"""RAFT optical-flow estimator — drop-in for SPyNet.

Same call signature as SPyNet: ``raft(img1, img2) -> flow [B,2,H,W]``.
Inputs are expected in [0,1] float (matching the old SPyNet wrapper).
Internally normalised to [-1,1] for torchvision RAFT and padded to a
multiple of 8 (RAFT requirement).

NOTE: RAFT-small's correlation pyramid requires inputs >=128 pixels
in each spatial dimension. Smaller inputs (e.g. 64x64 from CFR's 8x
downsample) produce NaN. The forward() method handles this by upsampling
small inputs to >=128, running RAFT, then scaling the flow back.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small

# pad multiple required by RAFT
_PAD_MULT = 8
# minimum spatial size for RAFT correlation pyramid (empirical threshold)
_MIN_SIZE = 128
# number of update iterations during inference (12 = torchvision default)
_RAFT_ITERS = 12


class RAFTFlowEstimator(nn.Module):
    """Pretrained RAFT-small wrapper that mimics SPyNet's interface.

    Why RAFT-small: 1.0M params vs RAFT-large 5.3M; +0.3-0.5dB EPE worse
    on Sintel but ~3x faster, fits 24GB GPU comfortably alongside SD2.1.
    """

    def __init__(self, pretrained: bool = True, small: bool = True,
                 checkpoint_path: str = None):
        super().__init__()
        if not small:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self.net = raft_large(weights=None)
            if pretrained:
                if checkpoint_path is not None:
                    self.net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                else:
                    ckpt = Raft_Large_Weights.DEFAULT
                    self.net = raft_large(weights=ckpt, progress=False)
        else:
            self.net = raft_small(weights=None)
            if pretrained:
                if checkpoint_path is not None:
                    self.net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                else:
                    import os
                    cache = os.path.expanduser('~/.cache/torch/hub/checkpoints/raft_small-8bb27295.pth')
                    if os.path.exists(cache):
                        self.net.load_state_dict(torch.load(cache, map_location='cpu'))
                    else:
                        from torchvision.models.optical_flow import Raft_Small_Weights
                        self.net = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
        self.net.eval()
        # RAFT dtype auto-detected in forward() — no forced .float() here
        for p in self.net.parameters():
            p.requires_grad = False

    @staticmethod
    def _pad_to_mult(x: torch.Tensor):
        _, _, h, w = x.shape
        ph = (_PAD_MULT - h % _PAD_MULT) % _PAD_MULT
        pw = (_PAD_MULT - w % _PAD_MULT) % _PAD_MULT
        if ph == 0 and pw == 0:
            return x, (0, 0, 0, 0)
        x_p = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x_p, (0, pw, 0, ph)

    @staticmethod
    def _unpad_flow(flow: torch.Tensor, pad, orig_h, orig_w):
        return flow[:, :, :orig_h, :orig_w]

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # SPyNet contract: img in [0,1], C in {1,3}.
        if img1.dim() != 4 or img2.dim() != 4:
            raise ValueError(f"expected [B,C,H,W], got {img1.shape}, {img2.shape}")
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        # RAFT (torchvision) expects [-1,1].
        model_dtype = next(self.net.parameters()).dtype
        x1 = (img1.clamp(0, 1) * 2.0 - 1.0).to(dtype=model_dtype)
        x2 = (img2.clamp(0, 1) * 2.0 - 1.0).to(dtype=model_dtype)

        _, _, h, w = x1.shape

        # RAFT-small correlation pyramid needs >=128 px per dim.
        # CFR feeds 64x64 after 8x downsample; SPyNet handles this
        # (pyramid-based) but RAFT produces NaN.  Upsample small inputs
        # to >=128, run RAFT, then scale flow back proportionally.
        if h < _MIN_SIZE or w < _MIN_SIZE:
            scale = max(_MIN_SIZE / h, _MIN_SIZE / w)
            new_h = int((h * scale + _PAD_MULT - 1) // _PAD_MULT * _PAD_MULT)
            new_w = int((w * scale + _PAD_MULT - 1) // _PAD_MULT * _PAD_MULT)
            x1u = F.interpolate(x1, size=(new_h, new_w), mode='bilinear',
                                align_corners=False)
            x2u = F.interpolate(x2, size=(new_h, new_w), mode='bilinear',
                                align_corners=False)
            with torch.no_grad():
                flow = self.net(x1u, x2u, num_flow_updates=_RAFT_ITERS)[-1]
            flow = F.interpolate(flow, size=(h, w), mode='bilinear',
                                 align_corners=False)
            flow[:, 0] *= w / new_w
            flow[:, 1] *= h / new_h
            return flow.to(img1.dtype)

        x1p, pad = self._pad_to_mult(x1)
        x2p, _ = self._pad_to_mult(x2)

        with torch.no_grad():
            flows = self.net(x1p, x2p, num_flow_updates=_RAFT_ITERS)
        flow = flows[-1]  # final refined flow
        flow = self._unpad_flow(flow, pad, h, w)
        return flow.to(img1.dtype)
