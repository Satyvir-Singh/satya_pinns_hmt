"""Ghia data loading, coordinate orientation, error calculations, and isolation
of the benchmark data from training."""
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cavity_pinn.benchmark import load_ghia_u, load_ghia_v, profile_errors
from cavity_pinn.evaluate import ghia_metrics
from cavity_pinn.model import CavityPINN

PKG = Path(__file__).resolve().parent.parent / "cavity_pinn"


def test_loading_and_orientation():
    gu, gv = load_ghia_u(), load_ghia_v()
    assert gu["xy"].shape == (17, 2) and gv["xy"].shape == (17, 2)
    assert np.all(gu["xy"][:, 0] == 0.5) and np.array_equal(gu["xy"][:, 1], gu["coord"])
    assert np.all(gv["xy"][:, 1] == 0.5) and np.array_equal(gv["xy"][:, 0], gv["coord"])
    u = dict(zip(gu["coord"], gu["ref"]))
    v = dict(zip(gv["coord"], gv["ref"]))
    assert u[1.0] == 1.0 and u[0.0] == 0.0 and v[1.0] == 0.0 and v[0.0] == 0.0
    assert u[0.5] == -0.20581 and u[0.9766] == 0.84123 and u[0.4531] == -0.21090
    assert v[0.5] == 0.05454 and v[0.8047] == -0.24533 and v[0.2344] == 0.17527
    assert v[0.9531] == -0.08864 and v[0.9453] == -0.10313
    # physical orientation: lid moves +x, clockwise primary vortex:
    # u>0 near lid, u<0 in lower half; v<0 near the right wall, v>0 near the left wall
    assert u[0.9531] > 0 and u[0.1719] < 0 and v[0.8594] < 0 and v[0.1563] > 0


def test_profile_errors():
    e = profile_errors(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
    assert e["rmse"] == pytest.approx(np.sqrt(5 / 3)) and e["max_abs"] == 2.0 and e["n"] == 3


class CoordCore(nn.Module):
    """u := y, v := x, p := 0 (in physical coords) to verify point placement."""

    def forward(self, z):
        x, y = (z[:, 0:1] + 1) / 2, (z[:, 1:2] + 1) / 2
        return torch.cat([y, x, 0 * x], 1)


def test_model_evaluated_directly_at_tabulated_coordinates():
    m = ghia_metrics(CavityPINN(core=CoordCore()), batch=5)
    gu, gv = m["u(x=0.5, y)"], m["v(x, y=0.5)"]
    np.testing.assert_allclose(gu["pred"], gu["coord"], atol=1e-15)
    np.testing.assert_allclose(gv["pred"], gv["coord"], atol=1e-15)
    err = np.array(gu["coord"]) - np.array(gu["ref"])
    assert gu["rmse"] == pytest.approx(np.sqrt(np.mean(err ** 2)), rel=1e-14)
    assert gu["max_abs"] == pytest.approx(np.abs(err).max(), rel=1e-14)


def test_training_code_does_not_touch_benchmark():
    for f in ("train.py", "model.py", "physics.py", "sampling.py", "config.py", "utils.py"):
        src = (PKG / f).read_text(encoding="utf-8").lower()
        assert "benchmark" not in src.replace("benchmark data are deliberately not imported", "")
        assert "ghia" not in src.replace("ghia benchmark data are deliberately not imported", "")
