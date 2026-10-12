"""Automatic first/second derivatives incl. the internal input scaling."""
import numpy as np
import pytest
import torch
from torch import nn

from cavity_pinn.model import CavityPINN
from cavity_pinn.physics import derivatives, vorticity


class AnalyticCore(nn.Module):
    """Outputs analytic functions of the *scaled* coordinates (xi, eta)."""

    def forward(self, z):
        xi, eta = z[:, 0:1], z[:, 1:2]
        return torch.cat([torch.sin(xi) * eta ** 2, torch.exp(0.5 * xi) * torch.cos(eta), xi ** 3 * eta], 1)


@pytest.mark.parametrize("lo,hi", [((0.0, 0.0), (1.0, 1.0)), ((-1.0, 2.0), (3.0, 2.5))])
def test_derivatives_through_scaling(lo, hi):
    model = CavityPINN(core=AnalyticCore(), lo=lo, hi=hi)
    g = torch.Generator().manual_seed(0)
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    xy = lo_t + (hi_t - lo_t) * torch.rand(200, 2, generator=g)
    d = derivatives(model, xy)
    a, b = 2.0 / (hi_t - lo_t)                       # d xi/dx, d eta/dy
    xi = (2 * (xy - lo_t) / (hi_t - lo_t) - 1)[:, 0:1]
    eta = (2 * (xy - lo_t) / (hi_t - lo_t) - 1)[:, 1:2]
    exact = dict(
        u_x=a * torch.cos(xi) * eta ** 2, u_y=b * 2 * torch.sin(xi) * eta,
        u_xx=-a ** 2 * torch.sin(xi) * eta ** 2, u_yy=b ** 2 * 2 * torch.sin(xi),
        v_x=a * 0.5 * torch.exp(0.5 * xi) * torch.cos(eta), v_y=-b * torch.exp(0.5 * xi) * torch.sin(eta),
        v_xx=a ** 2 * 0.25 * torch.exp(0.5 * xi) * torch.cos(eta),
        v_yy=-b ** 2 * torch.exp(0.5 * xi) * torch.cos(eta),
        p_x=a * 3 * xi ** 2 * eta, p_y=b * xi ** 3)
    for k, v in exact.items():
        torch.testing.assert_close(d[k], v, rtol=1e-12, atol=1e-12, msg=k)
    om = vorticity(model, xy)
    torch.testing.assert_close(om, exact["v_x"] - exact["u_y"], rtol=1e-12, atol=1e-12)


def test_network_derivatives_vs_finite_differences():
    torch.manual_seed(3)
    model = CavityPINN(width=16, depth=3)
    xy = torch.rand(50, 2) * 0.8 + 0.1
    d = derivatives(model, xy)
    h1, h2 = 1e-5, 1e-4
    for j, axis in enumerate("xy"):
        e = torch.zeros(1, 2)
        e[0, j] = 1.0
        with torch.no_grad():
            fp, fm, f0 = model(xy + h1 * e), model(xy - h1 * e), model(xy)
            fp2, fm2 = model(xy + h2 * e), model(xy - h2 * e)
        fd1 = (fp - fm) / (2 * h1)
        fd2 = (fp2 - 2 * f0 + fm2) / h2 ** 2
        np.testing.assert_allclose(d[f"u_{axis}"].detach(), fd1[:, 0:1], atol=1e-8)
        np.testing.assert_allclose(d[f"v_{axis}"].detach(), fd1[:, 1:2], atol=1e-8)
        np.testing.assert_allclose(d[f"p_{axis}"].detach(), fd1[:, 2:3], atol=1e-8)
        np.testing.assert_allclose(d[f"u_{axis}{axis}"].detach(), fd2[:, 0:1], atol=1e-5)
        np.testing.assert_allclose(d[f"v_{axis}{axis}"].detach(), fd2[:, 1:2], atol=1e-5)


def test_float64_and_xavier():
    model = CavityPINN(width=64, depth=6)
    assert all(p.dtype == torch.float64 for p in model.parameters())
    assert len(model.core.hidden) == 6 and model.core.out.out_features == 3
    w = model.core.hidden[1].weight
    assert abs(w.std().item() - (2.0 / 128) ** 0.5) < 0.02     # xavier-normal std
    assert torch.all(model.core.hidden[0].bias == 0)
