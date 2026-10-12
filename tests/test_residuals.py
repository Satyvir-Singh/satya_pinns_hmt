"""PDE residuals against fields with independently (hand-)derived residuals."""
import math

import torch
from torch import nn

from cavity_pinn.model import CavityPINN
from cavity_pinn.physics import ns_residuals

RE = 100.0


def _pts(n=300, seed=1):
    return torch.rand(n, 2, generator=torch.Generator().manual_seed(seed))


def test_kovasznay_exact_solution_has_zero_residual():
    """Kovasznay (1948) flow is an exact steady NS solution for the given Re."""
    lam = RE / 2 - math.sqrt(RE ** 2 / 4 + 4 * math.pi ** 2)

    def f(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        u = 1 - torch.exp(lam * x) * torch.cos(2 * math.pi * y)
        v = lam / (2 * math.pi) * torch.exp(lam * x) * torch.sin(2 * math.pi * y)
        p = 0.5 * (1 - torch.exp(2 * lam * x))
        return torch.cat([u, v, p], 1)

    r = ns_residuals(f, _pts(), RE)
    for k in ("rc", "ru", "rv"):
        assert r[k].abs().max() < 1e-12, k


def test_polynomial_manufactured_residuals():
    # u = x^2 y, v = -x y^2, p = x y  (divergence free)
    # r_u = x^3 y^2 + y - 2y/Re,  r_v = x^2 y^3 + x + 2x/Re   (derived by hand)
    def f(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return torch.cat([x ** 2 * y, -x * y ** 2, x * y], 1)

    xy = _pts()
    x, y = xy[:, 0:1], xy[:, 1:2]
    r = ns_residuals(f, xy, RE)
    torch.testing.assert_close(r["rc"], torch.zeros_like(x), atol=1e-14, rtol=0)
    torch.testing.assert_close(r["ru"], x ** 3 * y ** 2 + y - 2 * y / RE, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(r["rv"], x ** 2 * y ** 3 + x + 2 * x / RE, atol=1e-13, rtol=1e-13)


class TrigCore(nn.Module):
    """u = sin(pi x) cos(pi y), v = sin(pi y), p = 0, written in scaled coords."""

    def forward(self, z):
        x, y = (z[:, 0:1] + 1) / 2, (z[:, 1:2] + 1) / 2
        return torch.cat([torch.sin(math.pi * x) * torch.cos(math.pi * y),
                          torch.sin(math.pi * y), 0 * x], 1)


def test_trig_manufactured_residuals_through_network_scaling():
    model = CavityPINN(core=TrigCore())
    xy = _pts(seed=5)
    x, y = xy[:, 0:1], xy[:, 1:2]
    pi = math.pi
    s, c = torch.sin, torch.cos
    rc = pi * c(pi * y) * (c(pi * x) + 1)
    ru = (s(pi * x) * c(pi * y) * pi * c(pi * x) * c(pi * y)
          + s(pi * y) * (-pi * s(pi * x) * s(pi * y))
          + 2 * pi ** 2 * s(pi * x) * c(pi * y) / RE)
    rv = s(pi * y) * pi * c(pi * y) + pi ** 2 * s(pi * y) / RE
    r = ns_residuals(model, xy, RE)
    torch.testing.assert_close(r["rc"], rc, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(r["ru"], ru, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(r["rv"], rv, atol=1e-12, rtol=1e-12)


def test_eval_mode_residuals_match_training_mode():
    torch.manual_seed(0)
    model = CavityPINN(width=16, depth=2)
    xy = _pts(50)
    a = ns_residuals(model, xy, RE, create_graph=True)
    b = ns_residuals(model, xy, RE, create_graph=False)
    for k in a:
        torch.testing.assert_close(a[k].detach(), b[k].detach(), rtol=0, atol=0)
