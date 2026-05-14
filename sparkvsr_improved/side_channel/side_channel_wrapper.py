"""Combines DetailNet + GateNet into one trainable head that lives
after the (frozen) DLoRAL VAE-decode output.

Forward signature:
    y_final = y_coarse + alpha * d_t

where d_t = DetailNet(x_up, y_coarse) and
      alpha = GateNet(x_up, y_coarse).

Set ``return_aux=True`` during training to also get back the alpha
map (for sparsity loss) and the raw d_t (for diagnostic logging /
visualisation). At inference time omit it for the cheapest forward.
"""

from typing import Tuple, Dict, Optional, Union

import torch
import torch.nn as nn

from .detail_net import DetailNet
from .gate_net import GateNet


class SideChannelWrapper(nn.Module):
    def __init__(self, detail_net: Optional[DetailNet] = None,
                 gate_net: Optional[GateNet] = None):
        super().__init__()
        self.detail_net = detail_net if detail_net is not None else DetailNet()
        self.gate_net = gate_net if gate_net is not None else GateNet()

    def forward(self, x_up: torch.Tensor, y_coarse: torch.Tensor,
                *, return_aux: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        d_t = self.detail_net(x_up, y_coarse)
        alpha = self.gate_net(x_up, y_coarse)
        y_final = y_coarse + alpha * d_t
        if return_aux:
            return y_final, {"alpha": alpha, "d_t": d_t}
        return y_final

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
