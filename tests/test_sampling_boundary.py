"""Boundary targets, upper-corner exclusion and interior sampling distribution."""
import numpy as np
import pytest

from cavity_pinn.evaluate import boundary_eval_points
from cavity_pinn.sampling import (UPPER_CORNERS, WALLS, boundary_corner_report, interior_density,
                                  sample_boundary, sample_interior)


@pytest.fixture(scope="module")
def bnd():
    return sample_boundary(500, seed=101)


def test_boundary_targets(bnd):
    xy, uv = bnd["top"]
    assert np.all(xy[:, 1] == 1.0) and np.all(uv[:, 0] == 1.0) and np.all(uv[:, 1] == 0.0)
    for w, col, val in (("bottom", 1, 0.0), ("left", 0, 0.0), ("right", 0, 1.0)):
        xy, uv = bnd[w]
        assert np.all(xy[:, col] == val) and np.all(uv == 0.0), w
    assert all(bnd[w][0].shape == (500, 2) for w in WALLS)


def test_upper_corners_excluded_and_reported(bnd):
    for w in ("top", "left", "right"):
        xy = bnd[w][0]
        tang = xy[:, 0] if w == "top" else xy[:, 1]
        assert np.all((tang > 0) & (tang < 1)), w      # open segments
        assert not any(np.all(xy == c, axis=1).any() for c in UPPER_CORNERS)
    rep = boundary_corner_report(bnd)
    allxy = np.concatenate([bnd[w][0] for w in WALLS])
    d = np.linalg.norm(allxy[:, None] - UPPER_CORNERS[None], axis=2).min(0)
    assert rep["all_walls"]["upper_left"] == pytest.approx(d[0])
    assert rep["all_walls"]["upper_right"] == pytest.approx(d[1])
    assert 0 < d.min() < 0.01        # no silent finite exclusion zone
    # the constant-speed lid is not tapered: every top target is exactly 1
    assert np.all(bnd["top"][1][:, 0] == 1.0)


def test_evaluation_boundary_points_independent_and_open():
    ev = boundary_eval_points(4000, seed=192837465)
    tr = sample_boundary(500, seed=101)
    for w in WALLS:
        xy = ev[w][0]
        tang = xy[:, 0] if w in ("top", "bottom") else xy[:, 1]
        assert np.all((tang > 0) & (tang < 1))
        assert len(np.intersect1d(tang, tr[w][0][:, 0] if w in ("top", "bottom") else tr[w][0][:, 1])) == 0


def test_interior_sampling_distribution():
    n, a, dlt = 20000, 0.2, 0.05
    xy = sample_interior(n, a, dlt, seed=7)
    assert xy.shape == (n, 2) and np.all((xy > 0) & (xy < 1))
    # documented density integrates to one
    g = (np.arange(2000) + 0.5) / 2000
    X, Y = np.meshgrid(g, g)
    f = interior_density(np.c_[X.ravel(), Y.ravel()], a, dlt)
    assert f.mean() == pytest.approx(1.0, abs=2e-3)
    # empirical fraction in the 4 bands matches the density prediction
    band = (xy.min(1) < dlt) | (xy.max(1) > 1 - dlt)
    pred = (1 - a) * (1 - (1 - 2 * dlt) ** 2) + a
    assert band.mean() == pytest.approx(pred, abs=0.01)
    # deterministic given seed
    np.testing.assert_array_equal(xy, sample_interior(n, a, dlt, seed=7))
