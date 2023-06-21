"""
Multi modules for RGB-FIR
"""

import torch
import torch.nn as nn

__all__ = [
    "S2M_RGB",
    "S2M_FIR",
    "M2S",
    "Pass",
]


class S2M_RGB(nn.Module):
    """Multi-Channel Split Layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.unbind(x, 1)
        x = torch.stack(x[:3], dim=1)
        return x


class S2M_FIR(nn.Module):
    """Multi-Channel Split Layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.unbind(x, 1)
        x = torch.unsqueeze(x[3], dim=1)
        return x


class M2S(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Pass(nn.Module):
    """Set Point to path the input tensros"""

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return x
