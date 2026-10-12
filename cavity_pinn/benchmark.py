"""Ghia, Ghia & Shin (1982) Re = 100 centreline reference data and error metrics.

The data are used ONLY for post-training validation.  Nothing in this module is
imported by the training code (enforced by tests/test_benchmark.py).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TABLE1_U = DATA_DIR / "ghia1982_re100_table1_u_vertical_centerline.csv"
TABLE2_V = DATA_DIR / "ghia1982_re100_table2_v_horizontal_centerline.csv"


def read_commented_csv(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read a CSV whose leading lines starting with '#' are provenance comments."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.reader(lines))
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(c) for c in r] for r in rows[1:]], dtype=np.float64)
    return header, data


def load_ghia_u(path: str | Path = TABLE1_U) -> dict:
    """Table I: u along the vertical centreline x = 0.5.  Returns points (x, y)."""
    header, d = read_commented_csv(path)
    assert header == ["y", "u"], header
    y, u = d[:, 0], d[:, 1]
    xy = np.c_[np.full_like(y, 0.5), y]
    return dict(name="u(x=0.5, y)", coord_name="y", coord=y, xy=xy, ref=u, component=0)


def load_ghia_v(path: str | Path = TABLE2_V) -> dict:
    """Table II: v along the horizontal centreline y = 0.5.  Returns points (x, y)."""
    header, d = read_commented_csv(path)
    assert header == ["x", "v"], header
    x, v = d[:, 0], d[:, 1]
    xy = np.c_[x, np.full_like(x, 0.5)]
    return dict(name="v(x, y=0.5)", coord_name="x", coord=x, xy=xy, ref=v, component=1)


def profile_errors(pred: np.ndarray, ref: np.ndarray) -> dict:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))),
                max_abs=float(np.max(np.abs(err))),
                mean_abs=float(np.mean(np.abs(err))),
                n=int(err.size))
