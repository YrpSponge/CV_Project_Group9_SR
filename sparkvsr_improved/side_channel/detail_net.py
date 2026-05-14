"""Pixel-space residual head that recovers high-frequency detail
the SD VAE compresses away during DLoRAL's latent decode."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))


class DetailNet(nn.Module):
    """Predicts a per-pixel residual ``d_t`` from concatenated
    ``(x_up, y_coarse)``.

    Shape contract:
        x_up, y_coarse: (B, 3, H, W) in [0, 1] — must already be the
        same HR resolution.
        returns d_t: (B, 3, H, W) un-bounded (caller multiplies by
        gate alpha then adds to y_coarse).

    Init:
        Final conv is zero-initialised so ``d_t == 0`` at step 0,
        which means ``y_final == y_coarse`` and the frozen DLoRAL
        baseline is preserved at the very start of training.
    """

    def __init__(self, in_ch: int = 6, out_ch: int = 3,
                 base_ch: int = 128, num_blocks: int = 16):
        super().__init__()
        self.head = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.body = nn.ModuleList([_ResBlock(base_ch) for _ in range(num_blocks)])
        self.body_out = nn.Conv2d(base_ch, base_ch, 3, padding=1)
        self.tail = nn.Conv2d(base_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x_up: torch.Tensor, y_coarse: torch.Tensor) -> torch.Tensor:
        h0 = self.head(torch.cat([x_up, y_coarse], dim=1))
        h = h0
        for blk in self.body:
            h = blk(h)
        h = h0 + self.body_out(h)
        return self.tail(h)
