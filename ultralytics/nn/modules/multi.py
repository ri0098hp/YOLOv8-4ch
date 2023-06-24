"""
Multi modules for RGB-FIR
"""

import torch
import torch.nn as nn

from .conv import Conv

__all__ = [
    "S2M_RGB",
    "S2M_FIR",
    "M2S",
    "M2S_Conv",
    "M2S_Add",
    "Pass",
]


class S2M_RGB(nn.Module):
    """Multi-Channel Split Layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.unbind(x, 1)
        x = torch.stack(x[1:], dim=1)
        return x


class S2M_FIR(nn.Module):
    """Multi-Channel Split Layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.unbind(x, 1)
        x = torch.unsqueeze(x[0], dim=1)
        return x


class M2S(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Merge features with Concatenating"""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class M2S_Conv(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, c1, c2, k=1, s=1, dimension=1):  # ch_in, ch_out, kernel
        """Merge features with Depthwise Conv"""
        super().__init__()
        self.fuse = Conv(c1, c2, k, s)
        self.d = dimension

    def forward(self, x):
        return self.fuse(torch.cat(x, self.d))


class M2S_Add(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self):  # ch_in, ch_out, number
        """Merge maps with adding"""
        super().__init__()

    def forward(self, x):
        return sum(x)


class Pass(nn.Module):
    """Set Point to path the input tensros"""

    def forward(self, x):
        return x
