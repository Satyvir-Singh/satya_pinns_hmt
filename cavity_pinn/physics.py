"""Steady incompressible Navier-Stokes residuals via automatic differentiation.

Nondimensionalisation: length L (cavity width), velocity U (lid speed),
pressure rho*U^2, Re = U L / nu.

    r_c = u_x + v_y
    r_u = u u_x + v u_y + p_x - (u_xx + u_yy) / Re
    r_v = u v_x + v v_y + p_y - (v_xx + v_yy) / Re

All derivatives are taken w.r.t. the *physical* coordinates (x, y) that are
passed to ``fields_fn``; any internal input scaling is differentiated through.
"""
from __future__ import annotations

from typing import Callable

import torch

FieldsFn = Callable[[torch.Tensor], torch.Tensor]


def _grad(f: torch.Tensor, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                               create_graph=create_graph, retain_graph=True)[0]


def derivatives(fields_fn: FieldsFn, xy: torch.Tensor, create_graph: bool = True) -> dict:
    """Return u, v, p and their first/second derivatives needed by the residuals.

    ``create_graph=False`` is for evaluation only (no parameter gradients needed
    afterwards); first derivatives still need a graph for second derivatives.
    """
    if not xy.requires_grad:
        xy = xy.clone().requires_grad_(True)
    out = fields_fn(xy)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    du = _grad(u, xy, True)
    dv = _grad(v, xy, True)
    dp = _grad(p, xy, create_graph)
    u_x, u_y = du[:, 0:1], du[:, 1:2]
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]
    u_xx = _grad(u_x, xy, create_graph)[:, 0:1]
    u_yy = _grad(u_y, xy, create_graph)[:, 1:2]
    v_xx = _grad(v_x, xy, create_graph)[:, 0:1]
    v_yy = _grad(v_y, xy, create_graph)[:, 1:2]
    return dict(u=u, v=v, p=p, u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y,
                p_x=dp[:, 0:1], p_y=dp[:, 1:2],
                u_xx=u_xx, u_yy=u_yy, v_xx=v_xx, v_yy=v_yy)


def residuals_from_derivatives(d: dict, re: float) -> dict:
    rc = d["u_x"] + d["v_y"]
    ru = d["u"] * d["u_x"] + d["v"] * d["u_y"] + d["p_x"] - (d["u_xx"] + d["u_yy"]) / re
    rv = d["u"] * d["v_x"] + d["v"] * d["v_y"] + d["p_y"] - (d["v_xx"] + d["v_yy"]) / re
    return dict(rc=rc, ru=ru, rv=rv)


def ns_residuals(fields_fn: FieldsFn, xy: torch.Tensor, re: float,
                 create_graph: bool = True) -> dict:
    return residuals_from_derivatives(derivatives(fields_fn, xy, create_graph), re)


def vorticity(fields_fn: FieldsFn, xy: torch.Tensor) -> torch.Tensor:
    """omega = v_x - u_y computed by automatic differentiation."""
    xy = xy.clone().requires_grad_(True)
    out = fields_fn(xy)
    du = _grad(out[:, 0:1], xy, False)
    dv = _grad(out[:, 1:2], xy, False)
    return dv[:, 0:1] - du[:, 1:2]
