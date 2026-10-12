"""PINN training for the Re = 100 lid-driven cavity: Adam (refreshing interior
points) followed by full-batch L-BFGS on frozen points with frozen weights.

Usage:
    python -m cavity_pinn.train --config config/re100.yaml --seed 101
    python -m cavity_pinn.train --resume runs/re100_seed101/checkpoints/checkpoint_last.pt
    python -m cavity_pinn.train --resume <ckpt> --set lbfgs.iterations=8000   # extend budget

Ghia benchmark data are deliberately NOT imported here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .config import apply_overrides, load_config, save_config
from .model import build_model
from .physics import ns_residuals
from .sampling import WALLS, boundary_corner_report, sample_boundary, sample_interior, \
    upper_corner_distances
from .utils import (get_rng_states, hardware_info, peak_memory_mb, seed_everything,
                    select_device, set_rng_states)

torch.set_default_dtype(torch.float64)

COMPONENTS = ["continuity", "momentum_u", "momentum_v"] + [f"wall_{w}" for w in WALLS] + ["gauge"]


def interior_seed(seed: int, idx: int) -> int:
    return seed * 100_003 + idx


LBFGS_SET = 99_999  # index of the frozen interior set used by L-BFGS


class Trainer:
    def __init__(self, cfg: dict, seed: int, run_dir: str | Path, verbose: bool = True):
        self.cfg, self.seed, self.verbose = cfg, int(seed), verbose
        self.run_dir = Path(run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        tr = cfg["training"]
        if int(tr.get("num_threads", 0)) > 0:
            torch.set_num_threads(int(tr["num_threads"]))
        self.device = select_device(tr.get("device", "auto"))
        seed_everything(self.seed)
        self.model = build_model(cfg).to(self.device)
        self.re = float(cfg["physics"]["re"])
        w = cfg["weights"]
        self.weights = {k: float(w[k]) for k in ("continuity", "momentum_u", "momentum_v", "wall", "gauge")}
        self.gauge_xy = torch.tensor([cfg["physics"]["gauge_point"]], device=self.device)

        s = cfg["sampling"]
        bnd = sample_boundary(int(s["n_boundary_per_wall"]), seed=self.seed)
        self.corner_report = boundary_corner_report(bnd)
        self.Xb = torch.tensor(np.concatenate([bnd[w][0] for w in WALLS]), device=self.device)
        self.Tb = torch.tensor(np.concatenate([bnd[w][1] for w in WALLS]), device=self.device)
        self.wall_slices, i0 = {}, 0
        for wname in WALLS:
            n = bnd[wname][0].shape[0]
            self.wall_slices[wname] = slice(i0, i0 + n)
            i0 += n

        self.phase = "adam"
        self.adam_it = 0
        self.lbfgs_it = 0
        self.n_evals = 0          # optimizer loss+gradient evaluations
        self.n_diag_evals = 0     # extra loss evaluations for logging only
        self.refresh_idx = 0
        self.last_refresh_it = 0
        self.elapsed = 0.0
        self.history: list[dict] = []
        self.status = "running"
        self.set_interior(self.refresh_idx)
        self.optimizer = None
        self.scheduler = None
        self._make_optimizer()
        self.hw = hardware_info(self.device)

    # ------------------------------------------------------------------ setup
    def set_interior(self, idx: int) -> None:
        s = self.cfg["sampling"]
        xy = sample_interior(int(s["n_interior"]), float(s["near_wall_fraction"]),
                             float(s["near_wall_delta"]), interior_seed(self.seed, idx))
        self.interior_corner_dist = upper_corner_distances(xy)
        self.X = torch.tensor(xy, device=self.device)

    def _make_optimizer(self) -> None:
        if self.phase == "adam":
            a = self.cfg["adam"]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(a["lr"]))
            gamma = float(a["decay_rate"]) ** (1.0 / float(a["decay_steps"]))
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
        else:
            lb = self.cfg["lbfgs"]
            self.optimizer = torch.optim.LBFGS(
                self.model.parameters(), lr=1.0, max_iter=int(lb["max_iter_per_step"]),
                history_size=int(lb["history_size"]), tolerance_grad=float(lb["tolerance_grad"]),
                tolerance_change=float(lb["tolerance_change"]),
                line_search_fn=lb.get("line_search", "strong_wolfe"))
            self.scheduler = None

    # ------------------------------------------------------------------- loss
    def loss(self) -> tuple[torch.Tensor, dict]:
        r = ns_residuals(self.model, self.X, self.re)
        comps = dict(continuity=(r["rc"] ** 2).mean(),
                     momentum_u=(r["ru"] ** 2).mean(),
                     momentum_v=(r["rv"] ** 2).mean())
        out_b = self.model(self.Xb)[:, :2]
        sq = ((out_b - self.Tb) ** 2).sum(1)
        for wname, sl in self.wall_slices.items():
            comps[f"wall_{wname}"] = sq[sl].mean()
        comps["gauge"] = self.model(self.gauge_xy)[0, 2] ** 2
        w = self.weights
        total = (w["continuity"] * comps["continuity"] + w["momentum_u"] * comps["momentum_u"]
                 + w["momentum_v"] * comps["momentum_v"]
                 + w["wall"] * sum(comps[f"wall_{wn}"] for wn in WALLS)
                 + w["gauge"] * comps["gauge"])
        return total, comps

    # ---------------------------------------------------------------- logging
    def _record(self, total, comps, it_label: int, lr: float) -> dict:
        rec = dict(phase=self.phase, iteration=it_label, adam_it=self.adam_it, lbfgs_it=self.lbfgs_it,
                   n_evals=self.n_evals, n_diag_evals=self.n_diag_evals, lr=lr,
                   total=float(total), **{k: float(v) for k, v in comps.items()},
                   **{f"w_{k}": v for k, v in self.weights.items()},
                   elapsed_s=self.elapsed_now(), peak_mem_mb=peak_memory_mb(self.device),
                   refresh_idx=self.refresh_idx)
        self.history.append(rec)
        if self.verbose:
            print(f"[{self.phase:5s} {it_label:6d}] total={rec['total']:.3e} "
                  f"c={rec['continuity']:.2e} mu={rec['momentum_u']:.2e} mv={rec['momentum_v']:.2e} "
                  f"top={rec['wall_top']:.2e} bot={rec['wall_bottom']:.2e} "
                  f"lft={rec['wall_left']:.2e} rgt={rec['wall_right']:.2e} g={rec['gauge']:.1e} "
                  f"lr={lr:.2e} evals={self.n_evals} t={rec['elapsed_s']:.0f}s "
                  f"mem={rec['peak_mem_mb'] or float('nan'):.0f}MB", flush=True)
        return rec

    def elapsed_now(self) -> float:
        return self.elapsed + (time.perf_counter() - self._t0)

    def write_history(self) -> None:
        if not self.history:
            return
        keys = list(self.history[0].keys())
        with open(self.run_dir / "train_history.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(self.history)

    # ------------------------------------------------------------ checkpoints
    def state(self) -> dict:
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict() if self.scheduler else None,
                    config=self.cfg, seed=self.seed, phase=self.phase, adam_it=self.adam_it,
                    lbfgs_it=self.lbfgs_it, n_evals=self.n_evals, n_diag_evals=self.n_diag_evals,
                    refresh_idx=self.refresh_idx, last_refresh_it=self.last_refresh_it,
                    elapsed=self.elapsed_now(), history=self.history, status=self.status,
                    interior=self.X.cpu(), boundary=self.Xb.cpu(), rng=get_rng_states(),
                    corner_report=self.corner_report, hardware=self.hw,
                    torch_version=torch.__version__)

    def save_checkpoint(self, name: str = "checkpoint_last.pt") -> Path:
        path = self.run_dir / "checkpoints" / name
        tmp = path.with_suffix(".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, path)
        if self.cfg["training"].get("keep_checkpoints", False) and name == "checkpoint_last.pt":
            it = self.adam_it if self.phase == "adam" else self.lbfgs_it
            shutil.copyfile(path, path.with_name(f"snapshot_{self.phase}_{it:06d}.pt"))
        self.write_history()
        return path

    @classmethod
    def from_checkpoint(cls, path: str | Path, overrides: list[str] | None = None,
                        verbose: bool = True) -> "Trainer":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = apply_overrides(ck["config"], overrides)
        run_dir = Path(path).resolve().parent.parent
        tr = cls(cfg, ck["seed"], run_dir, verbose=verbose)
        tr.model.load_state_dict(ck["model"])
        for k in ("phase", "adam_it", "lbfgs_it", "n_evals", "n_diag_evals", "refresh_idx",
                  "last_refresh_it", "history", "status"):
            setattr(tr, k, ck[k])
        tr.elapsed = ck["elapsed"]
        tr.X = ck["interior"].to(tr.device)
        assert torch.equal(ck["boundary"], tr.Xb.cpu()), "boundary points not reproducible"
        tr._make_optimizer()
        tr.optimizer.load_state_dict(ck["optimizer"])
        if tr.scheduler is not None and ck["scheduler"] is not None:
            tr.scheduler.load_state_dict(ck["scheduler"])
        set_rng_states(ck["rng"])
        if tr.status in ("done", "lbfgs_converged") and overrides:
            tr.status = "running"  # budget extended on resume
        return tr

    # --------------------------------------------------------------- training
    def run(self) -> None:
        self._t0 = time.perf_counter()
        save_config(self.cfg, self.run_dir / "config.yaml")
        meta = dict(seed=self.seed, hardware=self.hw, corner_report_boundary_training=self.corner_report,
                    interior_initial_upper_corner_distance=self.interior_corner_dist)
        with open(self.run_dir / "run_info.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self.verbose:
            print(f"device={self.device} threads={torch.get_num_threads()} cpu={self.hw.get('cpu')}")
            print(f"nearest boundary training point to upper corners: {self.corner_report['all_walls']}")
        if self.phase == "adam":
            self._run_adam()
        if self.phase == "lbfgs":
            self._run_lbfgs()
        self.save_checkpoint("checkpoint_last.pt")
        self.save_checkpoint("checkpoint_final.pt")
        if self.verbose:
            print(f"finished: status={self.status} elapsed={self.elapsed_now():.0f}s evals={self.n_evals}")

    def _run_adam(self) -> None:
        a, s, tr = self.cfg["adam"], self.cfg["sampling"], self.cfg["training"]
        n_it, refresh = int(a["iterations"]), int(s.get("refresh_every", 0))
        log_every, ck_every = int(tr["log_every"]), int(tr["checkpoint_every"])
        while self.adam_it < n_it:
            if refresh and self.adam_it > 0 and self.adam_it % refresh == 0 \
                    and self.last_refresh_it != self.adam_it:
                self.refresh_idx += 1
                self.last_refresh_it = self.adam_it
                self.set_interior(self.refresh_idx)
            lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.zero_grad(set_to_none=True)
            total, comps = self.loss()
            if not torch.isfinite(total):
                self.status = "adam_nonfinite"
                print("non-finite loss in Adam; stopping", flush=True)
                return
            total.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.adam_it += 1
            self.n_evals += 1
            if self.adam_it % log_every == 0 or self.adam_it == 1 or self.adam_it == n_it:
                self._record(total.detach(), {k: v.detach() for k, v in comps.items()}, self.adam_it, lr)
            if self.adam_it % ck_every == 0:
                self.save_checkpoint()
        self.save_checkpoint("checkpoint_adam_end.pt")
        # switch to L-BFGS: freeze a dedicated interior set and the loss weights
        self.phase = "lbfgs"
        self.set_interior(LBFGS_SET)
        self._make_optimizer()
        self.save_checkpoint()

    def _diag_eval(self) -> tuple[torch.Tensor, dict]:
        """Loss at the current parameters (for logging; not an optimizer evaluation)."""
        total, comps = self.loss()
        self.n_diag_evals += 1
        return total.detach(), {k: v.detach() for k, v in comps.items()}

    def _run_lbfgs(self) -> None:
        lb, tr = self.cfg["lbfgs"], self.cfg["training"]
        n_it, per_step = int(lb["iterations"]), int(lb["max_iter_per_step"])
        ck_every = int(tr["checkpoint_every"])
        params = list(self.model.parameters())
        if self.lbfgs_it == 0:
            total, comps = self._diag_eval()
            self._record(total, comps, 0, 1.0)

        def closure():
            # deterministic: fixed points, fixed weights, no randomness
            self.optimizer.zero_grad(set_to_none=False)
            total, _ = self.loss()
            total.backward()
            self.n_evals += 1
            return total

        last_ck = self.lbfgs_it
        while self.lbfgs_it < n_it:
            max_iter = min(per_step, n_it - self.lbfgs_it)
            group = self.optimizer.param_groups[0]
            group["max_iter"], group["max_eval"] = max_iter, max_iter * 5 // 4
            st = self.optimizer.state.get(params[0], {})
            it0, ev0 = st.get("n_iter", 0), self.n_evals
            backup = [p.detach().clone() for p in params]
            self.optimizer.step(closure)
            st = self.optimizer.state[params[0]]
            done_it, done_ev = st["n_iter"] - it0, self.n_evals - ev0
            if not all(torch.isfinite(p).all() for p in params):
                with torch.no_grad():
                    for p, b in zip(params, backup):
                        p.copy_(b)
                self.status = "lbfgs_nonfinite_reverted"
                print("non-finite parameters after L-BFGS step; reverted and stopped", flush=True)
                break
            self.lbfgs_it += done_it
            total, comps = self._diag_eval()
            self._record(total, comps, self.lbfgs_it, 1.0)
            if self.lbfgs_it - last_ck >= ck_every:
                self.save_checkpoint()
                last_ck = self.lbfgs_it
            if done_it < max_iter and done_ev < group["max_eval"]:
                self.status = "lbfgs_converged"   # tolerance_grad / tolerance_change met
                print(f"L-BFGS stopped by tolerance after {self.lbfgs_it} iterations", flush=True)
                return
        if self.status == "running":
            self.status = "done"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/re100.yaml")
    ap.add_argument("--seed", type=int, default=None, help="default: experiment.baseline_seed")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    args = ap.parse_args(argv)
    if args.resume:
        tr = Trainer.from_checkpoint(args.resume, args.set)
    else:
        cfg = load_config(args.config, args.set)
        seed = args.seed if args.seed is not None else int(cfg["experiment"]["baseline_seed"])
        run_dir = args.run_dir or Path(cfg["experiment"]["output_dir"]) / \
            f"{cfg['experiment']['name']}{args.tag}_seed{seed}"
        tr = Trainer(cfg, seed, run_dir)
    tr.run()


if __name__ == "__main__":
    main()
