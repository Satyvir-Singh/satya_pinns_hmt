"""Aggregate the three seed runs and the sensitivity run into tables and figures.

Usage:
    python -m cavity_pinn.summarize [--seeds 101 202 303] [--baseline 101]
           [--sens runs/re100_sens_w96_n20k_seed101] [--out runs/summary]
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

from .benchmark import load_ghia_u, load_ghia_v  # noqa: E402
from .evaluate import predict  # noqa: E402
from .model import build_model  # noqa: E402

torch.set_default_dtype(torch.float64)


def load_run(d: Path):
    mp, cp = d / "eval" / "metrics.json", d / "checkpoints" / "checkpoint_final.pt"
    if not (mp.exists() and cp.exists()):
        return None
    m = json.loads(mp.read_text())
    ck = torch.load(cp, map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    return dict(dir=d, metrics=m, model=model, config=ck["config"])


def centerlines(model, s):
    u = predict(model, np.c_[np.full_like(s, 0.5), s])[:, 0]
    v = predict(model, np.c_[s, np.full_like(s, 0.5)])[:, 1]
    return u, v


def fmt(x):
    return "n/a" if x is None else f"{x:.3e}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--name", default="re100")
    ap.add_argument("--seeds", type=int, nargs="*", default=[101, 202, 303])
    ap.add_argument("--baseline", type=int, default=101)
    ap.add_argument("--sens", default="runs/re100_sens_w96_n20k_seed101")
    ap.add_argument("--out", default="runs/summary")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {s: load_run(Path(a.runs_root) / f"{a.name}_seed{s}") for s in a.seeds}
    sens = load_run(Path(a.sens))
    done = {s: r for s, r in runs.items() if r is not None}
    lines = ["# Re = 100 cavity PINN: multi-seed summary", "",
             f"Baseline seed (chosen in advance): {a.baseline}. Seeds: {a.seeds}. "
             f"Completed: {sorted(done)}; missing: {sorted(set(a.seeds) - set(done))}.", ""]

    # ---- acceptance table
    keys = None
    table = {}
    for s, r in done.items():
        acc = r["metrics"]["acceptance"]
        keys = keys or list(acc)
        table[s] = acc
    if sens:
        table["sens"] = sens["metrics"]["acceptance"]
        keys = keys or list(table["sens"])
    if keys:
        cols = [str(s) for s in done] + (["sens(w96,n20k,s101)"] if sens else [])
        lines += ["## Acceptance criteria (project targets, not Ghia criteria)", "",
                  "| criterion | " + " | ".join(cols) + " | seed mean ± std |",
                  "|---|" + "---|" * (len(cols) + 1)]
        for k in keys:
            vals = [table[s][k]["value"] for s in done]
            cells = [f"{table[s][k]['value']:.3e} {table[s][k]['result']}" for s in done]
            if sens:
                cells.append(f"{table['sens'][k]['value']:.3e} {table['sens'][k]['result']}")
            ms = f"{np.mean(vals):.3e} ± {np.std(vals, ddof=1):.1e}" if len(vals) > 1 else "n/a"
            lines.append(f"| {k} | " + " | ".join(cells) + f" | {ms} |")
        lines.append("")

    # ---- run info
    lines += ["## Runs", "", "| run | status | Adam it | L-BFGS it | optimizer evals | train time [h] | "
              "peak mem [MB] | final total loss | pde rms excl. r<0.05 upper-corner disks (c, mu, mv) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, r in [(str(s), r) for s, r in done.items()] + ([("sens", sens)] if sens else []):
        m = r["metrics"]
        p = m["pde"]
        lines.append(f"| {name} | {m['status']} | {m['adam_iterations']} | {m['lbfgs_iterations']} | "
                     f"{m['optimizer_evals']} | {m['train_elapsed_s'] / 3600:.2f} | {m.get('peak_mem_mb', 0):.0f} | "
                     f"{m['final_training_loss']['total']:.3e} | "
                     + ", ".join(fmt(p[k]['rms_excl_upper_corner_disks']) for k in
                                 ('continuity', 'momentum_u', 'momentum_v')) + " |")
    lines.append("")

    # ---- boundary near-corner breakdown
    lines += ["## Boundary errors: near-corner (within r of a wall end point) vs rest", "",
              "| run | wall | comp | RMS all | RMS near-corner | max near-corner | RMS rest | max rest |",
              "|---|---|---|---|---|---|---|---|"]
    for name, r in [(str(s), r) for s, r in done.items()] + ([("sens", sens)] if sens else []):
        b = r["metrics"]["boundary"]
        for w in ("top", "bottom", "left", "right"):
            for c in ("u", "v"):
                d = b[w][c]
                lines.append(f"| {name} | {w} | {c} | {d['rms']:.2e} | {fmt(d['near_corner_rms'])} | "
                             f"{fmt(d['near_corner_max_abs'])} | {d['rest_rms']:.2e} | {d['rest_max_abs']:.2e} |")
    lines.append("")

    # ---- centreline figure & csv
    gu, gv = load_ghia_u(), load_ghia_v()
    s = np.linspace(0, 1, 401)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    curves = {}
    for i, (name, r) in enumerate([(f"seed {k}", v) for k, v in done.items()]
                                  + ([("sens w96 n20k (seed 101)", sens)] if sens else [])):
        u, v = centerlines(r["model"], s)
        curves[name] = (u, v)
        ls = "--" if name.startswith("sens") else "-"
        axs[0].plot(u, s, ls, lw=1.2, label=name, color=f"C{i}")
        axs[1].plot(s, v, ls, lw=1.2, label=name, color=f"C{i}")
    axs[0].plot(gu["ref"], gu["coord"], "ko", mfc="none", label="Ghia et al. (1982)")
    axs[1].plot(gv["coord"], gv["ref"], "ko", mfc="none", label="Ghia et al. (1982)")
    axs[0].set(xlabel="u", ylabel="y", title="u along x = 0.5")
    axs[1].set(xlabel="x", ylabel="v", title="v along y = 0.5")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ghia_all_runs.png", dpi=150)
    plt.close(fig)

    rows = []
    allr = [(f"seed{k}", v) for k, v in done.items()] + ([("sens", sens)] if sens else [])
    for prof, comp in ((gu, 0), (gv, 1)):
        preds = {n: predict(r["model"], prof["xy"])[:, comp] for n, r in allr}
        for j, c in enumerate(prof["coord"]):
            row = dict(profile=prof["name"], coord=f"{c:.4f}", ghia=f"{prof['ref'][j]:.5f}")
            for n, p in preds.items():
                row[f"pred_{n}"] = f"{p[j]:.6f}"
                row[f"err_{n}"] = f"{p[j] - prof['ref'][j]:.6f}"
            seed_preds = [preds[f"seed{k}"][j] for k in done]
            if len(seed_preds) > 1:
                row["seed_std"] = f"{np.std(seed_preds, ddof=1):.6f}"
            rows.append(row)
    if rows:
        with open(out / "centerline_predictions_all_runs.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    # ---- seed variability and sensitivity
    if len(done) > 1:
        U = np.array([curves[f"seed {k}"][0] for k in done])
        V = np.array([curves[f"seed {k}"][1] for k in done])
        lines += ["## Seed variability (dense centrelines, 401 points)", "",
                  f"- max over line of the std. dev. across seeds: u {U.std(0, ddof=1).max():.3e}, "
                  f"v {V.std(0, ddof=1).max():.3e}",
                  f"- max pairwise |difference| between seeds: u {np.ptp(U, 0).max():.3e}, "
                  f"v {np.ptp(V, 0).max():.3e}", ""]
    if sens and a.baseline in done:
        ub, vb = curves[f"seed {a.baseline}"]
        us, vs = curves["sens w96 n20k (seed 101)"]
        mb, msn = done[a.baseline]["metrics"], sens["metrics"]
        lines += ["## Sensitivity study (seed 101: width 64 → 96, n_interior 10k → 20k; all else identical)", "",
                  "This is a sensitivity check, not a proof of convergence.", "",
                  f"- dense-centreline change: max |Δu| = {np.abs(us - ub).max():.3e}, "
                  f"RMS Δu = {np.sqrt(np.mean((us - ub) ** 2)):.3e}; max |Δv| = {np.abs(vs - vb).max():.3e}, "
                  f"RMS Δv = {np.sqrt(np.mean((vs - vb) ** 2)):.3e}", "",
                  "| metric | baseline (s101) | sensitivity | change |", "|---|---|---|---|"]
        for k in mb["acceptance"]:
            x0, x1 = mb["acceptance"][k]["value"], msn["acceptance"][k]["value"]
            lines.append(f"| {k} | {x0:.3e} | {x1:.3e} | {x1 / x0 if x0 else float('nan'):.2f}× |")
        lines.append("")
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(s, us - ub)
        axs[0].set(xlabel="y", ylabel="Δu", title="u(0.5, y): sensitivity − baseline")
        axs[1].plot(s, vs - vb)
        axs[1].set(xlabel="x", ylabel="Δv", title="v(x, 0.5): sensitivity − baseline")
        for ax in axs:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "sensitivity_centerline_change.png", dpi=140)
        plt.close(fig)

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
