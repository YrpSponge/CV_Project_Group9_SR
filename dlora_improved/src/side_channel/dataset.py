"""REDS multi-frame dataset with on-the-fly Real-ESRGAN degradation.

For side-channel training: returns (HR center frame, T LR-bicubic-up frames).
"""
from __future__ import annotations
import glob
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_THIS_DIR))
from datasets.realesrgan import RealESRGAN_degradation


class REDSMultiFrame(Dataset):
    def __init__(
        self,
        root: str,
        deg_yaml: str,
        hr_size: int = 512,
        num_frames: int = 2,
    ):
        self.root = root
        self.hr_size = hr_size
        self.num_frames = num_frames
        clip_dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
        self.samples: List[Tuple[str, int, List[str]]] = []
        for cdir in clip_dirs:
            frames = sorted(glob.glob(os.path.join(cdir, "*.png")))
            if len(frames) < num_frames:
                continue
            for s in range(0, len(frames) - num_frames + 1):
                self.samples.append((cdir, s, frames))
        if not self.samples:
            raise RuntimeError(f"No REDS clips with >= {num_frames} frames under {root}")
        self.degradation = RealESRGAN_degradation(deg_yaml, device="cpu")

    def __len__(self) -> int:
        return len(self.samples)

    def _shared_crop_box(self, w: int, h: int) -> Tuple[int, int]:
        x = random.randint(0, max(0, w - self.hr_size))
        y = random.randint(0, max(0, h - self.hr_size))
        return x, y

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cdir, start, frames = self.samples[idx]
        ref = Image.open(frames[start]).convert("RGB")
        w, h = ref.size
        if w < self.hr_size or h < self.hr_size:
            scale = self.hr_size / min(w, h) + 1e-3
            ref = ref.resize((int(w * scale) + 1, int(h * scale) + 1), Image.BICUBIC)
            w, h = ref.size
        cx, cy = self._shared_crop_box(w, h)

        hr_list, lr_up_list = [], []
        for k in range(self.num_frames):
            img = Image.open(frames[start + k]).convert("RGB")
            iw, ih = img.size
            if iw < self.hr_size or ih < self.hr_size:
                scale = self.hr_size / min(iw, ih) + 1e-3
                img = img.resize((int(iw * scale) + 1, int(ih * scale) + 1), Image.BICUBIC)
                iw, ih = img.size
            cx_k = min(cx, iw - self.hr_size)
            cy_k = min(cy, ih - self.hr_size)
            img = img.crop((cx_k, cy_k, cx_k + self.hr_size, cy_k + self.hr_size))
            np_hr = np.asarray(img).astype(np.float32) / 255.0
            hr_t, lr_t = self.degradation.degrade_process(np_hr, resize_bak=True)
            hr_list.append(hr_t.squeeze(0) if hr_t.dim() == 4 else hr_t)
            lr_up_list.append(lr_t.squeeze(0) if lr_t.dim() == 4 else lr_t)

        hr = torch.stack(hr_list, dim=0).clamp(0, 1).float()
        lr_up = torch.stack(lr_up_list, dim=0).clamp(0, 1).float()
        return {"hr": hr, "lr_up": lr_up}
