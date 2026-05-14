"""Train side-channel modules (DetailNet + GateNet) on top of frozen DLoRAL.

Pipeline per step:
  HR  --Real-ESRGAN-->  LR  --bicubic_up-->  x_up
  x_up (T frames)  --frozen Generator_eval-->  y_coarse
  y_final = y_coarse + alpha(x_up_c, y_coarse) * d_t(x_up_c, y_coarse)
  loss   = L1(y_final, HR) + lpips_w * LPIPS(y_final, HR)

Multi-GPU: launch with `accelerate launch --num_processes=N src/train_side_channel.py --config ...`
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)
sys.path.append(os.path.dirname(_THIS_DIR))

from accelerate import Accelerator
from accelerate.utils import set_seed

from side_channel import DetailNet, GateNet, SideChannelWrapper
from side_channel.dataset import REDSMultiFrame
from DLoRAL_model import Generator_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dloral_args(cfg: dict) -> SimpleNamespace:
    d = cfg["dloral"]
    return SimpleNamespace(
        pretrained_path=d["pretrained_path"],
        pretrained_model_path=d["pretrained_model_path"],
        pretrained_model_name_or_path=d["pretrained_model_path"],
        vae_encoder_tiled_size=d.get("vae_encoder_tiled_size", 4096),
        vae_decoder_tiled_size=d.get("vae_decoder_tiled_size", 224),
        latent_tiled_size=d.get("latent_tiled_size", 96),
        latent_tiled_overlap=d.get("latent_tiled_overlap", 32),
        load_cfr=d.get("load_cfr", True),
        merge_and_unload_lora=False,
        mixed_precision=d.get("mixed_precision", "fp16"),
        process_size=d.get("process_size", 512),
    )


def compute_uncertainty_map(lr_up_gray: torch.Tensor) -> torch.Tensor:
    """lr_up_gray: [B, T, 1, H, W] -> [B, T-1, 1, H/8, W/8]."""
    b, t, _, h, w = lr_up_gray.shape
    if t < 2:
        return torch.zeros(b, 1, 1, h // 8, w // 8, device=lr_up_gray.device)
    out = []
    for i in range(1, t):
        pair = torch.stack([lr_up_gray[:, i], lr_up_gray[:, i - 1]], dim=1)
        var = pair.var(dim=1)
        thr = var.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
        mask = (var >= thr).float()
        mask = F.interpolate(mask, size=(h // 8, w // 8), mode="bilinear", align_corners=False)
        out.append(mask)
    return torch.stack(out, dim=1)


@torch.no_grad()
def dloral_y_coarse(model: Generator_eval, x_up: torch.Tensor, weight_dtype) -> torch.Tensor:
    """x_up: [B, T, 3, H, W] in [0, 1] -> y_coarse: [B, 3, H, W] in [0, 1]."""
    c_t = (x_up * 2.0 - 1.0).to(dtype=weight_dtype)
    gray = 0.299 * x_up[:, :, 0:1] + 0.587 * x_up[:, :, 1:2] + 0.114 * x_up[:, :, 2:3]
    um = compute_uncertainty_map(gray)
    out, *_ = model(stages=1, c_t=c_t, uncertainty_map=um.to(dtype=weight_dtype),
                    prompt="", weight_dtype=weight_dtype)
    return (out.float() * 0.5 + 0.5).clamp(0, 1)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    accelerator = Accelerator()
    set_seed(cfg.get("seed", 42))
    is_main = accelerator.is_main_process
    device = accelerator.device

    # ---------- DLoRAL (frozen, per-rank copy) ----------
    dloral_args = build_dloral_args(cfg)
    dloral = Generator_eval(dloral_args)
    dloral.unet.set_adapter([
        "default_encoder_quality", "default_decoder_quality", "default_others_quality",
        "default_encoder_consistency", "default_decoder_consistency", "default_others_consistency",
    ])
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        dloral_args.mixed_precision
    ]
    dloral.vae = dloral.vae.to(dtype=weight_dtype)
    dloral.unet = dloral.unet.to(dtype=weight_dtype)
    dloral.cfr_main_net = dloral.cfr_main_net.to(dtype=weight_dtype)
    for p in dloral.parameters():
        p.requires_grad = False
    dloral.eval()

    # ---------- side-channel ----------
    wrapper = SideChannelWrapper(
        DetailNet(**cfg.get("detail_net", {})),
        GateNet(**cfg.get("gate_net", {})),
    )
    if is_main:
        n = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
        print(f"[info] side-channel trainable params: {n/1e6:.2f}M  | world_size={accelerator.num_processes}")

    # ---------- data ----------
    ds_cfg = cfg["data"]
    dataset = REDSMultiFrame(
        root=ds_cfg["reds_root"],
        deg_yaml=ds_cfg["deg_yaml"],
        hr_size=ds_cfg.get("hr_size", 512),
        num_frames=ds_cfg.get("num_frames", 2),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )
    if is_main:
        print(f"[info] dataset samples: {len(dataset)}")

    # ---------- losses & optim ----------
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False
    optim = torch.optim.AdamW(wrapper.parameters(), lr=cfg.get("lr", 1e-4),
                              weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.get("max_steps", 5000), eta_min=cfg.get("min_lr", 1e-6)
    )
    lpips_w = cfg.get("lambda_lpips", 0.5)

    out_dir = cfg["output_dir"]
    if is_main:
        os.makedirs(out_dir, exist_ok=True)

    wrapper, optim, loader, sched = accelerator.prepare(wrapper, optim, loader, sched)

    # ---------- loop ----------
    step = 0
    max_steps = cfg.get("max_steps", 5000)
    log_every = cfg.get("log_every", 25)
    save_every = cfg.get("save_every", 500)
    t0 = time.time()
    wrapper.train()
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            hr = batch["hr"].to(device, non_blocking=True)
            lr_up = batch["lr_up"].to(device, non_blocking=True)

            y_coarse = dloral_y_coarse(dloral, lr_up, weight_dtype)
            x_up_c = lr_up[:, -1]  # match DLoRAL output (window last frame)
            hr_c = hr[:, -1]  # match DLoRAL output (window last frame)

            y_final, aux = wrapper(x_up_c, y_coarse.detach(), return_aux=True)
            l1 = F.l1_loss(y_final, hr_c)
            lp = lpips_fn(y_final * 2 - 1, hr_c * 2 - 1).mean()
            loss = l1 + lpips_w * lp

            optim.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(wrapper.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1

            if is_main and step % log_every == 0:
                dt = time.time() - t0
                print(
                    f"[{step:>5}/{max_steps}] loss={loss.item():.4f} "
                    f"l1={l1.item():.4f} lpips={lp.item():.4f} "
                    f"alpha={aux['alpha'].mean().item():.3f} "
                    f"d_t={aux['d_t'].abs().mean().item():.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} "
                    f"({step/max(dt,1e-3):.2f} it/s)",
                    flush=True,
                )
            if step % save_every == 0 or step == max_steps:
                accelerator.wait_for_everyone()
                if is_main:
                    unwrapped = accelerator.unwrap_model(wrapper)
                    ckpt = {
                        "step": step,
                        "wrapper": unwrapped.state_dict(),
                        "optim": optim.state_dict(),
                        "cfg": cfg,
                    }
                    path = os.path.join(out_dir, f"sidechannel_step{step:06d}.pt")
                    torch.save(ckpt, path)
                    print(f"[ckpt] saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
