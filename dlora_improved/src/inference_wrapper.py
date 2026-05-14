"""Clean inference API for DLoRAL VSR (W4+SC: CFR-RAFT + SideChannel).

Usage:
    from src.inference_wrapper import DLoRALInferenceWrapper

    model = DLoRALInferenceWrapper(
        pretrained_path="runs/cfr_raft_finetune/.../model_52001.pkl",
        flow_estimator="raft",
        sidechannel_ckpt="preset/models/side_channel/sidechannel_step005000.pt",
    )
    outputs = model(frames)  # frames: list of np.ndarray (H,W,3) or PIL.Image
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

_CWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)

from src.DLoRAL_model import Generator_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from src.side_channel import SideChannelWrapper


def _compute_frame_diff_mask(frames):
    var = frames.var(dim=0)
    threshold = var.mean().item()
    mask = torch.where(var >= threshold, var, torch.zeros_like(var))
    return torch.where(mask == 0, mask, torch.ones_like(mask))


def _to_numpy(t):
    arr = (t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr


DEFAULT_CONFIG = dict(
    pretrained_model_path="preset_models/stable-diffusion-2-1-base",
    ram_path=None,
    ram_ft_path=None,
    stages=None,
    mixed_precision="fp16",
    align_method="adain",
    process_size=512,
    upscale=4,
    seed=42,
    vae_decoder_tiled_size=224,
    vae_encoder_tiled_size=1024,
    latent_tiled_size=96,
    latent_tiled_overlap=32,
    merge_and_unload_lora=False,
    sidechannel_alpha_scale=1.0,
    prompt="",
    save_prompts=False,
    output_dir="/tmp/dloral_out",
)


class DLoRALInferenceWrapper:

    def __init__(
        self,
        pretrained_path,
        flow_estimator="raft",
        sidechannel_ckpt=None,
        **overrides,
    ):
        cfg = {**DEFAULT_CONFIG, **overrides}
        cfg["pretrained_path"] = pretrained_path
        cfg["flow_estimator"] = flow_estimator
        cfg["sidechannel_ckpt"] = sidechannel_ckpt
        cfg["load_cfr"] = True
        args = SimpleNamespace(**cfg)
        self.args = args

        self._dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.mixed_precision]

        self.model = Generator_eval(args)
        self.model.set_eval()
        self.model.vae = self.model.vae.to(dtype=self._dtype)
        self.model.unet = self.model.unet.to(dtype=self._dtype)
        self.model.cfr_main_net = self.model.cfr_main_net.to(dtype=self._dtype)

        if args.stages == 0:
            self.model.unet.set_adapter([
                "default_encoder_consistency", "default_decoder_consistency", "default_others_consistency"
            ])
        else:
            self.model.unet.set_adapter([
                "default_encoder_quality", "default_decoder_quality", "default_others_quality",
                "default_encoder_consistency", "default_decoder_consistency", "default_others_consistency",
            ])

        self.sidechannel = None
        if args.sidechannel_ckpt is not None:
            self.sidechannel = SideChannelWrapper().cuda().eval()
            ckpt = torch.load(args.sidechannel_ckpt, map_location="cuda")
            sd = ckpt["wrapper"] if isinstance(ckpt, dict) and "wrapper" in ckpt else ckpt
            self.sidechannel.load_state_dict(sd, strict=True)
            for p in self.sidechannel.parameters():
                p.requires_grad_(False)

        self._ram = None

    def _get_ram(self):
        if self._ram is not None:
            return self._ram
        from ram.models.ram_lora import ram
        from ram import inference_ram as inference
        m = ram(pretrained=self.args.ram_path, pretrained_condition=self.args.ram_ft_path,
                image_size=384, vit="swin_l")
        m.eval()
        m.to("cuda", dtype=self._dtype)
        self._ram = (m, inference)
        return self._ram

    def _generate_prompt(self, first_frame):
        model, infer_fn = self._get_ram()
        t = TF.to_tensor(first_frame).unsqueeze(0).cuda()
        t = T.Resize((384, 384))(t)
        t = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t)
        t = t.to(dtype=self._dtype)
        captions = infer_fn(t, model)
        return f"{captions[0]}, {self.args.prompt},"

    @torch.no_grad()
    def __call__(self, frames, prompt=None):
        if len(frames) < 2:
            raise ValueError(f"Need at least 2 frames, got {len(frames)}")

        # --- convert to PIL ---
        input_pils = []
        for f in frames:
            if isinstance(f, np.ndarray):
                pil = Image.fromarray(f.astype(np.uint8))
            else:
                pil = f.convert("RGB") if f.mode != "RGB" else f
            input_pils.append(pil)

        # --- preprocess ---
        rscale = self.args.upscale
        ori_w, ori_h = input_pils[0].size
        resize_flag = False
        processed = []
        pils_gray = []

        for pil in input_pils:
            w, h = pil.size
            if w < self.args.process_size // rscale or h < self.args.process_size // rscale:
                scale = (self.args.process_size // rscale) / min(w, h)
                pil = pil.resize((int(scale * w), int(scale * h)))
                resize_flag = True
            pil = pil.resize((pil.size[0] * rscale, pil.size[1] * rscale))
            new_w = pil.width - pil.width % 8
            new_h = pil.height - pil.height % 8
            pil = pil.resize((new_w, new_h), Image.LANCZOS)
            processed.append(pil)
            g = TF.to_tensor(pil.convert("L"))
            g = nn.functional.interpolate(g.unsqueeze(0), scale_factor=0.125).squeeze(0)
            pils_gray.append(g)

        validation_prompt = prompt or self._generate_prompt(processed[0])
        n = len(processed)

        outputs = {}
        for i in range(0, n, 1):
            if i + 1 >= n:
                end = n - i
            else:
                end = 2

            win_frames = []
            win_grays = []
            for j in range(end):
                idx = i + j
                if idx < 0 or idx >= n:
                    continue
                win_frames.append(TF.to_tensor(processed[idx]))
                win_grays.append(pils_gray[idx])

            input_t = torch.stack(win_frames, dim=0)
            if input_t.shape[0] == 1:
                break
            gray_t = torch.stack(win_grays, dim=0)

            umaps = []
            for fi in range(input_t.shape[0]):
                if fi != 0:
                    umaps.append(_compute_frame_diff_mask(gray_t))
            if not umaps:
                break
            umap = torch.stack(umaps)

            c_t = input_t.unsqueeze(0).cuda() * 2 - 1
            c_t = c_t.to(dtype=self._dtype)

            out_img, _, _, _, _ = self.model(
                stages=self.args.stages,
                c_t=c_t,
                uncertainty_map=umap.unsqueeze(0).cuda(),
                prompt=validation_prompt,
                weight_dtype=self._dtype,
            )

            if self.sidechannel is not None:
                yc = (out_img.float() * 0.5 + 0.5).clamp(0, 1)
                xu = (c_t[:, -1].float() * 0.5 + 0.5).clamp(0, 1)
                yf, aux = self.sidechannel(xu, yc, return_aux=True)
                if abs(self.args.sidechannel_alpha_scale - 1.0) > 1e-6:
                    yf = yc + self.args.sidechannel_alpha_scale * aux["alpha"] * aux["d_t"]
                out_img = (yf * 2 - 1).clamp(-1, 1).to(dtype=self._dtype)

            frame_t = (out_img[0].cpu() * 0.5 + 0.5).clamp(0, 1)
            out_pil = T.ToPILImage()(frame_t)

            src_idx = i + 1
            src_idx = max(0, min(src_idx, n - 1))
            source_pil = processed[src_idx]

            if self.args.align_method == "adain":
                out_pil = adain_color_fix(target=out_pil, source=source_pil)
            elif self.args.align_method == "wavelet":
                out_pil = wavelet_color_fix(target=out_pil, source=source_pil)

            if resize_flag:
                nw = int(self.args.upscale * ori_w)
                nh = int(self.args.upscale * ori_h)
                out_pil = out_pil.resize((nw, nh), Image.BICUBIC)

            outputs[src_idx] = out_pil
            torch.cuda.empty_cache()

        result = []
        for idx in range(n):
            if idx in outputs:
                result.append(_to_numpy(TF.to_tensor(outputs[idx])))
            elif 1 in outputs:
                result.append(_to_numpy(TF.to_tensor(outputs[1])))
            else:
                nearest = min(outputs.keys(), key=lambda k: abs(k - idx))
                result.append(_to_numpy(TF.to_tensor(outputs[nearest])))

        return result


def upscale_single(model, image, prompt=None):
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image.astype(np.uint8))
    else:
        pil = image.convert("RGB") if image.mode != "RGB" else image
    outputs = model([pil, pil], prompt=prompt)
    return outputs[-1]
