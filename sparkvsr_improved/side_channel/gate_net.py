"""Per-pixel gate that decides where DetailNet's residual is allowed
to contribute. A small symmetric U-Net producing alpha in [0, 1]
at full HR.

Bias init at +``init_open_logit`` makes alpha start near 1 so the
gradient flowing into DetailNet is not throttled at step 0; combined
with DetailNet's zero-init tail this still yields y_final = y_coarse
initially. The gate then *learns to close* on regions where the
residual hurts, rather than *learning to open* from a saturated zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class GateNet(nn.Module):
    def __init__(self, in_ch: int = 6, base_ch: int = 64,
                 init_open_logit: float = 2.0):
        super().__init__()
        self.enc1 = _DoubleConv(in_ch, base_ch)
        self.enc2 = _DoubleConv(base_ch, base_ch * 2)
        self.enc3 = _DoubleConv(base_ch * 2, base_ch * 4)
        self.pool = nn.AvgPool2d(2)
        self.dec2 = _DoubleConv(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = _DoubleConv(base_ch * 2 + base_ch, base_ch)
        self.tail = nn.Conv2d(base_ch, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.constant_(self.tail.bias, float(init_open_logit))

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x_up: torch.Tensor, y_coarse: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_up, y_coarse], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self._up(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2), e1], dim=1))
        return torch.sigmoid(self.tail(d1))
