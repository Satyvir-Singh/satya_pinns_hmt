"""Collocation and boundary sampling for the unit lid-driven cavity.

Interior sampling distribution (documented, see README "Sampling"):
    a fraction (1 - alpha) of points is a scrambled Sobol sequence on (0,1)^2
    (density 1), and a fraction alpha is a near-wall layer: a wall is chosen
    uniformly at random, the wall-normal distance d ~ U(0, delta] and the
    tangential coordinate s ~ U(0, 1).  The resulting sampling density is

        f(x, y) = (1 - alpha) + alpha / (4 delta) * sum_w 1[dist_w(x, y) < delta].

    A sample mean of r^2 over these points therefore approximates the
    *weighted* integral  int r^2 f dA,  not the uniform-domain mean of r^2.

Boundary sampling: each wall gets its own scrambled Sobol points along the wall.
The top lid and the left/right walls are sampled on the *open* segments, so the
two upper corners (0,1) and (1,1), where the prescribed velocity is
discontinuous, are never used.  No finite exclusion zone is applied: the
nearest-sample distances to the upper corners are recorded instead.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import qmc

WALLS = ("top", "bottom", "left", "right")
UPPER_CORNERS = np.array([[0.0, 1.0], [1.0, 1.0]])


def sobol(n: int, d: int, seed: int) -> np.ndarray:
    with warnings.catch_warnings():  # n is generally not a power of two
        warnings.simplefilter("ignore", UserWarning)
        return qmc.Sobol(d, scramble=True, seed=seed).random(n)


def _open_unit(n: int, seed: int) -> np.ndarray:
    """n Sobol points in the open interval (0, 1) (exact endpoints rejected)."""
    s = sobol(n + 8, 1, seed)[:, 0]
    s = s[(s > 0.0) & (s < 1.0)][:n]
    assert s.size == n
    return s


def sample_interior(n: int, near_wall_frac: float, delta: float, seed: int) -> np.ndarray:
    n_nw = int(round(n * near_wall_frac))
    n_u = n - n_nw
    pts = [sobol(n_u, 2, seed)]
    if n_nw:
        rng = np.random.default_rng(seed)
        wall = rng.integers(0, 4, n_nw)
        d = delta * (1.0 - rng.random(n_nw))      # (0, delta]
        s = _open_unit(n_nw, seed + 7919)          # tangential coordinate
        rng.shuffle(s)
        xy = np.empty((n_nw, 2))
        xy[wall == 0] = np.c_[s, 1.0 - d][wall == 0]   # near top
        xy[wall == 1] = np.c_[s, d][wall == 1]         # near bottom
        xy[wall == 2] = np.c_[d, s][wall == 2]         # near left
        xy[wall == 3] = np.c_[1.0 - d, s][wall == 3]   # near right
        pts.append(xy)
    xy = np.concatenate(pts, 0)
    # keep strictly inside the open domain
    xy = np.clip(xy, 1e-12, 1.0 - 1e-12)
    return xy


def interior_density(xy: np.ndarray, near_wall_frac: float, delta: float) -> np.ndarray:
    """Analytic sampling density f(x, y) of ``sample_interior`` (integrates to 1)."""
    x, y = xy[:, 0], xy[:, 1]
    count = (y > 1 - delta).astype(float) + (y < delta) + (x < delta) + (x > 1 - delta)
    return (1 - near_wall_frac) + near_wall_frac / (4 * delta) * count


def wall_points(wall: str, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map tangential coordinates s in (0,1) to wall points and (u, v) targets."""
    n = s.size
    if wall == "top":
        xy = np.c_[s, np.ones(n)]
        uv = np.c_[np.ones(n), np.zeros(n)]
    elif wall == "bottom":
        xy = np.c_[s, np.zeros(n)]
        uv = np.zeros((n, 2))
    elif wall == "left":
        xy = np.c_[np.zeros(n), s]
        uv = np.zeros((n, 2))
    elif wall == "right":
        xy = np.c_[np.ones(n), s]
        uv = np.zeros((n, 2))
    else:
        raise ValueError(wall)
    return xy, uv


def sample_boundary(n_per_wall: int, seed: int) -> dict:
    """Training boundary points per wall on open segments (corners excluded)."""
    out = {}
    for k, wall in enumerate(WALLS):
        s = _open_unit(n_per_wall, seed + 104729 * (k + 1))
        out[wall] = wall_points(wall, s)
    return out


def upper_corner_distances(xy: np.ndarray) -> dict:
    """Minimum distance from any point in ``xy`` to each upper corner."""
    return {name: float(np.min(np.linalg.norm(xy - c, axis=1)))
            for name, c in zip(("upper_left", "upper_right"), UPPER_CORNERS)}


def boundary_corner_report(bnd: dict) -> dict:
    allxy = np.concatenate([bnd[w][0] for w in WALLS], 0)
    rep = {"all_walls": upper_corner_distances(allxy)}
    for w in WALLS:
        rep[w] = upper_corner_distances(bnd[w][0])
    return rep
