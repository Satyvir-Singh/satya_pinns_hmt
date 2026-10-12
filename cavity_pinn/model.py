"""Fully connected PINN for (x, y) -> (u, v, p) on the unit square.

Inputs are given in physical dimensionless coordinates (x, y) in [0, 1]^2 and
mapped *inside* ``forward`` to [-1, 1]^2.  Because the affine map is part of the
autograd graph, derivatives taken w.r.t. the physical input automatically
include the chain-rule factor d(xi)/dx = 2 / (hi - lo).
"""
from __future__ import annotations

import torch
from torch import nn


class InputScaling(nn.Module):
    """Affine map [lo, hi]^d -> [-1, 1]^d (differentiable, part of the graph)."""

    def __init__(self, lo=(0.0, 0.0), hi=(1.0, 1.0)):
        super().__init__()
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float64))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float64))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return 2.0 * (xy - self.lo) / (self.hi - self.lo) - 1.0


class MLP(nn.Module):
    """tanh MLP with Xavier-normal weights and zero biases."""

    def __init__(self, n_in: int, n_out: int, width: int, depth: int):
        super().__init__()
        dims = [n_in] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], n_out)
        for lin in list(self.hidden) + [self.out]:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for lin in self.hidden:
            z = torch.tanh(lin(z))
        return self.out(z)


class CavityPINN(nn.Module):
    """Network u, v, p = N(scale(x, y)).  ``core`` can be replaced in tests."""

    def __init__(self, width: int = 64, depth: int = 6, core: nn.Module | None = None,
                 lo=(0.0, 0.0), hi=(1.0, 1.0)):
        super().__init__()
        self.scale = InputScaling(lo, hi)
        self.core = core if core is not None else MLP(2, 3, width, depth)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.core(self.scale(xy))


def build_model(cfg: dict) -> CavityPINN:
    m = cfg["model"]
    return CavityPINN(width=int(m["width"]), depth=int(m["depth"])).to(torch.float64)
