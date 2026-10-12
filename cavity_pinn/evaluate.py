"""Post-training evaluation: Ghia comparison, held-out PDE residuals, boundary
errors, pressure gauge, acceptance criteria, figures and structured-field export.

Usage:
    python -m cavity_pinn.evaluate --checkpoint runs/re100_seed101/checkpoints/checkpoint_final.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .benchmark import load_ghia_u, load_ghia_v, profile_errors  # noqa: E402
from .config import apply_overrides  # noqa: E402
from .model import build_model  # noqa: E402
from .physics import ns_residuals, vorticity  # noqa: E402
from .sampling import UPPER_CORNERS, WALLS, wall_points  # noqa: E402

torch.set_default_dtype(torch.float64)

# Predeclared project acceptance targets (NOT criteria from Ghia et al.).
TARGETS = dict(centerline_rmse=0.01, centerline_max_abs=0.03, pde_rms=1e-3,
               wall_rms=0.01, gauge_abs=1e-4)


# ----------------------------------------------------------------- batching
def predict(model, xy: np.ndarray, batch: int = 4096) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(xy), batch):
            out.append(model(torch.as_tensor(xy[i:i + batch], dtype=torch.float64)).numpy())
    return np.concatenate(out, 0)


def residuals_batched(model, xy: np.ndarray, re: float, batch: int = 4096) -> dict:
    acc = {"rc": [], "ru": [], "rv": []}
    for i in range(0, len(xy), batch):
        x = torch.as_tensor(xy[i:i + batch], dtype=torch.float64)
        r = ns_residuals(model, x, re, create_graph=False)
        for k in acc:
            acc[k].append(r[k].detach().numpy()[:, 0])
    return {k: np.concatenate(v) for k, v in acc.items()}


def vorticity_batched(model, xy: np.ndarray, batch: int = 4096) -> np.ndarray:
    out = []
    for i in range(0, len(xy), batch):
        out.append(vorticity(model, torch.as_tensor(xy[i:i + batch], dtype=torch.float64))
                   .detach().numpy()[:, 0])
    return np.concatenate(out)


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


# ------------------------------------------------------------------ metrics
def ghia_metrics(model, batch: int) -> dict:
    res = {}
    for prof in (load_ghia_u(), load_ghia_v()):
        pred = predict(model, prof["xy"], batch)[:, prof["component"]]
        res[prof["name"]] = dict(coord_name=prof["coord_name"], coord=prof["coord"].tolist(),
                                 ref=prof["ref"].tolist(), pred=pred.tolist(),
                                 err=(pred - prof["ref"]).tolist(), **profile_errors(pred, prof["ref"]))
    return res


def heldout_points(n: int, seed: int) -> np.ndarray:
    """Fixed, uniformly distributed held-out interior points (open unit square)."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    xy = xy[(xy > 0).all(1) & (xy < 1).all(1)]
    assert len(xy) == n
    return xy


def pde_metrics(model, re: float, n: int, seed: int, batch: int, corner_r: float) -> dict:
    xy = heldout_points(n, seed)
    r = residuals_batched(model, xy, re, batch)
    dmin = np.min(np.linalg.norm(xy[:, None, :] - UPPER_CORNERS[None], axis=2), axis=1)
    far = dmin >= corner_r
    out = dict(n_points=n, seed=seed, nearest_point_to_upper_corner=float(dmin.min()))
    for k, name in (("rc", "continuity"), ("ru", "momentum_u"), ("rv", "momentum_v")):
        out[name] = dict(rms=rms(r[k]), max_abs=float(np.abs(r[k]).max()),
                         rms_excl_upper_corner_disks=rms(r[k][far]),
                         rms_within_upper_corner_disks=rms(r[k][~far]) if (~far).any() else None)
    out["corner_disk_radius_diagnostic_only"] = corner_r
    out["n_within_upper_corner_disks"] = int((~far).sum())
    return out


def boundary_eval_points(n: int, seed: int) -> dict:
    """Independent uniform-random boundary points on the open wall segments."""
    rng = np.random.default_rng(seed)
    out = {}
    for w in WALLS:
        s = rng.random(n + 8)
        s = s[(s > 0) & (s < 1)][:n]
        out[w] = wall_points(w, np.sort(s))
    return out


def boundary_metrics(model, n: int, seed: int, batch: int, corner_r: float) -> dict:
    pts = boundary_eval_points(n, seed)
    out = {}
    for w in WALLS:
        xy, target = pts[w]
        err = predict(model, xy, batch)[:, :2] - target
        # wall end points (corners) of this wall; upper ones are the singular corners
        ends = {"top": [(0, 1), (1, 1)], "bottom": [(0, 0), (1, 0)],
                "left": [(0, 0), (0, 1)], "right": [(1, 0), (1, 1)]}[w]
        dend = np.min(np.stack([np.linalg.norm(xy - np.array(e, float), axis=1) for e in ends]), 0)
        dup = np.min(np.linalg.norm(xy[:, None, :] - UPPER_CORNERS[None], axis=2), axis=1)
        near = dend < corner_r
        rec = dict(n=len(xy), nearest_to_upper_corner=float(dup.min()))
        for j, comp in enumerate(("u", "v")):
            e = err[:, j]
            rec[comp] = dict(rms=rms(e), max_abs=float(np.abs(e).max()),
                             near_corner_rms=rms(e[near]) if near.any() else None,
                             near_corner_max_abs=float(np.abs(e[near]).max()) if near.any() else None,
                             rest_rms=rms(e[~near]), rest_max_abs=float(np.abs(e[~near]).max()))
        rec["n_near_corner"] = int(near.sum())
        out[w] = rec
    out["near_corner_radius"] = corner_r
    return out


def acceptance(m: dict) -> dict:
    t, acc = TARGETS, {}
    for name, key in (("u(x=0.5, y)", "u"), ("v(x, y=0.5)", "v")):
        g = m["ghia"][name]
        acc[f"centerline_{key}_rmse<={t['centerline_rmse']}"] = (g["rmse"], g["rmse"] <= t["centerline_rmse"])
        acc[f"centerline_{key}_maxabs<={t['centerline_max_abs']}"] = (g["max_abs"], g["max_abs"] <= t["centerline_max_abs"])
    for k in ("continuity", "momentum_u", "momentum_v"):
        v = m["pde"][k]["rms"]
        acc[f"pde_{k}_rms<={t['pde_rms']:g}"] = (v, v <= t["pde_rms"])
    for w in WALLS:
        for c in ("u", "v"):
            v = m["boundary"][w][c]["rms"]
            acc[f"wall_{w}_{c}_rms<={t['wall_rms']}"] = (v, v <= t["wall_rms"])
    g = m["gauge"]["abs_error"]
    acc[f"gauge_abs<={t['gauge_abs']:g}"] = (g, g <= t["gauge_abs"])
    return {k: dict(value=v, result="PASS" if ok else "FAIL") for k, (v, ok) in acc.items()}


# ------------------------------------------------------------------ figures
def plot_ghia(m: dict, out: Path, label: str = "PINN", model=None, batch=4096) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    s = np.linspace(0, 1, 401)
    gu, gv = m["ghia"]["u(x=0.5, y)"], m["ghia"]["v(x, y=0.5)"]
    if model is not None:
        pu = predict(model, np.c_[np.full_like(s, 0.5), s], batch)[:, 0]
        pv = predict(model, np.c_[s, np.full_like(s, 0.5)], batch)[:, 1]
        axs[0].plot(pu, s, "-", color="C0", label=label)
        axs[1].plot(s, pv, "-", color="C0", label=label)
    axs[0].plot(gu["ref"], gu["coord"], "o", mfc="none", color="k", label="Ghia et al. (1982)")
    axs[1].plot(gv["coord"], gv["ref"], "o", mfc="none", color="k", label="Ghia et al. (1982)")
    axs[0].set(xlabel="u", ylabel="y", title=f"u at x = 0.5 (RMSE {gu['rmse']:.2e}, max {gu['max_abs']:.2e})")
    axs[1].set(xlabel="x", ylabel="v", title=f"v at y = 0.5 (RMSE {gv['rmse']:.2e}, max {gv['max_abs']:.2e})")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_fields(F: dict, out_dir: Path) -> None:
    n = F["n"]
    X, Y = F["x"].reshape(n, n), F["y"].reshape(n, n)
    flds = dict(u=F["u"], v=F["v"], p=F["p"], speed=np.hypot(F["u"], F["v"]), vorticity=F["omega"])
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, (name, val) in zip(axs.flat, flds.items()):
        Z = val.reshape(n, n).copy()
        Z[F["corner"].reshape(n, n)] = np.nan  # exact upper corners not shown
        if name == "vorticity":
            lim = np.nanpercentile(np.abs(Z), 99)
            cs = ax.contourf(X, Y, np.clip(Z, -lim, lim), 41, cmap="RdBu_r")
            ax.set_title("vorticity ω = v_x − u_y (clipped at 99th pct |ω|)")
        else:
            cs = ax.contourf(X, Y, Z, 41, cmap="viridis" if name in ("speed",) else "RdBu_r")
            ax.set_title(name)
        fig.colorbar(cs, ax=ax)
        ax.set_aspect("equal")
    ax = axs.flat[5]
    sp = np.hypot(F["u"], F["v"]).reshape(n, n)
    ax.streamplot(X[0], Y[:, 0], F["u"].reshape(n, n), F["v"].reshape(n, n), density=2.0,
                  color=sp, cmap="viridis", linewidth=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), title="streamlines")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "fields_contours.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.streamplot(X[0], Y[:, 0], F["u"].reshape(n, n), F["v"].reshape(n, n), density=2.5,
                  color=sp, cmap="viridis", linewidth=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title="Velocity streamlines, Re = 100")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "streamlines.png", dpi=150)
    plt.close(fig)


def plot_history(history: list[dict], out: Path) -> None:
    if not history:
        return
    keys = ["continuity", "momentum_u", "momentum_v", "wall_top", "wall_bottom", "wall_left",
            "wall_right", "gauge", "total"]
    ev = np.array([h["n_evals"] for h in history])
    fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    n_adam = max([h["n_evals"] for h in history if h["phase"] == "adam"], default=None)
    for ax, k in zip(axs.flat, keys):
        ax.semilogy(ev, [max(h[k], 1e-20) for h in history], lw=1)
        if n_adam is not None:
            ax.axvline(n_adam, color="gray", ls="--", lw=0.8)
        ax.set_title(k + (" (weighted sum)" if k == "total" else " (unweighted)"))
        ax.grid(alpha=0.3)
    for ax in axs[-1]:
        ax.set_xlabel("optimizer loss evaluations (dashed: Adam → L-BFGS)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_residual_maps(model, re: float, n: int, batch: int, out: Path) -> dict:
    """|r| on the interior nodes of an n x n grid (boundary nodes, hence the exact
    upper corners, are excluded)."""
    s = np.linspace(0, 1, n)[1:-1]
    X, Y = np.meshgrid(s, s)
    r = residuals_batched(model, np.c_[X.ravel(), Y.ravel()], re, batch)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    stats = {}
    for ax, (k, name) in zip(axs, (("rc", "continuity"), ("ru", "x-momentum"), ("rv", "y-momentum"))):
        Z = np.log10(np.abs(r[k]).reshape(X.shape) + 1e-16)
        cs = ax.contourf(X, Y, Z, levels=np.linspace(-6, 1, 29), extend="both", cmap="magma")
        fig.colorbar(cs, ax=ax, label="log10 |r|")
        ax.set(title=f"{name} residual (grid RMS {rms(r[k]):.2e})", xlabel="x", ylabel="y")
        ax.set_aspect("equal")
        stats[name] = rms(r[k])
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return stats


# ------------------------------------------------------------------- export
def field_grid(model, n: int, batch: int) -> dict:
    s = np.linspace(0, 1, n)
    X, Y = np.meshgrid(s, s)          # row index j <-> y, column i <-> x
    xy = np.c_[X.ravel(), Y.ravel()]
    uvp = predict(model, xy, batch)
    om = vorticity_batched(model, xy, batch)
    corner = np.isclose(xy[:, 1], 1.0) & (np.isclose(xy[:, 0], 0.0) | np.isclose(xy[:, 0], 1.0))
    return dict(n=n, x=xy[:, 0], y=xy[:, 1], u=uvp[:, 0], v=uvp[:, 1], p=uvp[:, 2], omega=om,
                corner=corner)


CORNER_NOTE = ("Values are raw network outputs at every node, including boundary nodes. "
               "The two exact upper corners (0,1),(1,1) have an undefined (discontinuous) prescribed "
               "velocity; they are flagged upper_corner=1, were never trained on, and are excluded "
               "from all quantitative diagnostics.")


def export_fields(F: dict, out_dir: Path, stem: str = "fields") -> list[Path]:
    n = F["n"]
    names = ["x", "y", "u", "v", "p", "omega"]
    cols = np.c_[[F[k] for k in names]].T
    flag = F["corner"].astype(int)
    paths = []
    p = out_dir / f"{stem}_{n}x{n}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Re=100 lid-driven cavity PINN; {n}x{n} uniform grid, x fastest. {CORNER_NOTE}\n")
        wr = csv.writer(f)
        wr.writerow(names + ["upper_corner"])
        for row, fl in zip(cols, flag):
            wr.writerow([f"{v:.10e}" for v in row] + [fl])
    paths.append(p)
    p = out_dir / f"{stem}_{n}x{n}.plt"
    with open(p, "w", encoding="utf-8") as f:
        f.write('TITLE = "Re=100 lid-driven cavity PINN"\n')
        f.write("# " + CORNER_NOTE + "\n")
        f.write('VARIABLES = "x" "y" "u" "v" "p" "omega" "upper_corner"\n')
        f.write(f'ZONE T="PINN", I={n}, J={n}, DATAPACKING=POINT\n')
        for row, fl in zip(cols, flag):
            f.write(" ".join(f"{v:.10e}" for v in row) + f" {fl}\n")
    paths.append(p)
    p = out_dir / f"{stem}_{n}x{n}.vtk"
    with open(p, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Re=100 cavity PINN; upper_corner=1 flags exact upper corners (excluded from diagnostics)\n")
        f.write(f"ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS {n} {n} 1\nORIGIN 0 0 0\n")
        f.write(f"SPACING {1 / (n - 1):.12g} {1 / (n - 1):.12g} 1\nPOINT_DATA {n * n}\n")
        for k in ("u", "v", "p", "omega"):
            f.write(f"SCALARS {k} double 1\nLOOKUP_TABLE default\n")
            f.write("\n".join(f"{v:.10e}" for v in F[k]) + "\n")
        f.write("SCALARS upper_corner int 1\nLOOKUP_TABLE default\n")
        f.write("\n".join(str(v) for v in flag) + "\n")
        f.write("VECTORS velocity double\n")
        f.write("\n".join(f"{a:.10e} {b:.10e} 0" for a, b in zip(F["u"], F["v"])) + "\n")
    paths.append(p)
    return paths


def write_ghia_csv(m: dict, out_dir: Path) -> None:
    for name, fn, cname, vname in (("u(x=0.5, y)", "ghia_u_x0.5.csv", "y", "u"),
                                   ("v(x, y=0.5)", "ghia_v_y0.5.csv", "x", "v")):
        g = m["ghia"][name]
        with open(out_dir / fn, "w", newline="", encoding="utf-8") as f:
            f.write("# Reference: Ghia, Ghia & Shin (1982) Re=100 (transcription unverified vs original; "
                    "see data/PROVENANCE.md). PINN evaluated directly at tabulated coordinates.\n")
            wr = csv.writer(f)
            wr.writerow([cname, f"{vname}_ghia", f"{vname}_pinn", "error_pinn_minus_ghia", "abs_error"])
            for c, r, p, e in zip(g["coord"], g["ref"], g["pred"], g["err"]):
                wr.writerow([f"{c:.4f}", f"{r:.5f}", f"{p:.8f}", f"{e:.8f}", f"{abs(e):.8f}"])


# --------------------------------------------------------------------- main
def evaluate_checkpoint(ckpt_path: str | Path, out_dir: str | Path | None = None,
                        overrides: list[str] | None = None, figures: bool = True,
                        export: bool = True) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = apply_overrides(ck["config"], overrides)
    ev = cfg["evaluation"]
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    out_dir = Path(out_dir) if out_dir else Path(ckpt_path).resolve().parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    re, batch, cr = float(cfg["physics"]["re"]), int(ev["batch_size"]), float(ev["near_corner_radius"])
    m = dict(checkpoint=str(ckpt_path), seed=ck["seed"], phase=ck["phase"], status=ck["status"],
             adam_iterations=ck["adam_it"], lbfgs_iterations=ck["lbfgs_it"], optimizer_evals=ck["n_evals"],
             train_elapsed_s=ck["elapsed"], hardware=ck.get("hardware"),
             model=cfg["model"], sampling=cfg["sampling"],
             corner_report_boundary_training=ck.get("corner_report"))
    if ck["history"]:
        last = ck["history"][-1]
        m["final_training_loss"] = {k: last[k] for k in ("total", "continuity", "momentum_u", "momentum_v",
                                                         "wall_top", "wall_bottom", "wall_left",
                                                         "wall_right", "gauge")}
        m["peak_mem_mb"] = max((h["peak_mem_mb"] or 0) for h in ck["history"])
    m["ghia"] = ghia_metrics(model, batch)
    m["pde"] = pde_metrics(model, re, int(ev["heldout_interior"]), int(ev["heldout_seed"]), batch, cr)
    m["boundary"] = boundary_metrics(model, int(ev["boundary_per_wall"]), int(ev["boundary_seed"]), batch, cr)
    gp = np.array([cfg["physics"]["gauge_point"]])
    m["gauge"] = dict(point=gp[0].tolist(), p=float(predict(model, gp)[0, 2]),
                      abs_error=float(abs(predict(model, gp)[0, 2])))
    m["acceptance"] = acceptance(m)
    m["all_pass"] = all(v["result"] == "PASS" for v in m["acceptance"].values())
    write_ghia_csv(m, out_dir)
    if figures:
        plot_ghia(m, out_dir / "ghia_comparison.png", model=model, batch=batch)
        plot_history(ck["history"], out_dir / "loss_history.png")
        m["residual_grid_rms"] = plot_residual_maps(model, re, int(ev["grid_n"]), batch,
                                                    out_dir / "residual_maps.png")
    if export or figures:
        F = field_grid(model, int(ev["grid_n"]), batch)
        if figures:
            plot_fields(F, out_dir)
        if export:
            m["exports"] = [str(p) for p in export_fields(F, out_dir)]
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    return m


def print_summary(m: dict) -> None:
    print(f"seed {m['seed']}  status={m['status']}  adam={m['adam_iterations']} lbfgs={m['lbfgs_iterations']} "
          f"evals={m['optimizer_evals']}  train_time={m['train_elapsed_s']:.0f}s")
    for k, v in m["acceptance"].items():
        print(f"  {v['result']}  {k:40s} {v['value']:.3e}")
    for k in ("continuity", "momentum_u", "momentum_v"):
        d = m["pde"][k]
        print(f"  pde {k:11s} rms={d['rms']:.2e} max={d['max_abs']:.2e} "
              f"rms(excl. r<{m['pde']['corner_disk_radius_diagnostic_only']} of upper corners)="
              f"{d['rms_excl_upper_corner_disks']:.2e}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="evaluation overrides, e.g. evaluation.grid_n=301")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args(argv)
    m = evaluate_checkpoint(args.checkpoint, args.out_dir, args.set, not args.no_figures, not args.no_export)
    print_summary(m)


if __name__ == "__main__":
    main()
