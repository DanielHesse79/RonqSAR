"""Placeholder for optional GNN model."""
from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch_geometric
except Exception:  # pragma: no cover
    torch = None


class SimpleGIN(nn.Module):  # type: ignore
    def __init__(self):
        super().__init__()
        if torch is None:
            raise ImportError("PyTorch Geometric not installed")
        # Placeholder network
        self.linear = nn.Linear(10, 1)

    def forward(self, data):
        return self.linear(data.x).mean(dim=1)
