# PINN for the steady 2-D lid-driven cavity at Re = 100

A physics-informed neural network (PyTorch, float64) for the steady incompressible lid-driven
square cavity at Re = 100. It is validated independently against the centreline tables of
Ghia, Ghia & Shin (1982). Results are in [VALIDATION_REPORT.md](VALIDATION_REPORT.md), and data
provenance is in [data/PROVENANCE.md](data/PROVENANCE.md).

Scope: this single case only. There is no temperature, no other Reynolds number and no
parameterised network.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q          # verification tests (~25 s)
```

On Linux or macOS, use `.venv/bin/python`. The device is chosen automatically with
`training.device: auto`, which picks CUDA if available and otherwise the CPU. The chosen device
is printed at start-up and stored in `runs/<run>/run_info.json`.

## Commands

```bash
# smoke run (minutes)
python -m cavity_pinn.train --config config/re100.yaml --seed 101 --tag _smoke \
       --set adam.iterations=150 lbfgs.iterations=60 lbfgs.max_iter_per_step=20

# one full run (seed 101 = predeclared baseline)
python -m cavity_pinn.train --config config/re100.yaml --seed 101

# resume an interrupted run (exactly reproduces the uninterrupted trajectory)
python -m cavity_pinn.train --resume runs/re100_seed101/checkpoints/checkpoint_last.pt
# resume and extend the budget
python -m cavity_pinn.train --resume runs/re100_seed101/checkpoints/checkpoint_last.pt --set lbfgs.iterations=8000

# evaluation: metrics, PASS/FAIL table, figures, field export
python -m cavity_pinn.evaluate --checkpoint runs/re100_seed101/checkpoints/checkpoint_final.pt
python -m cavity_pinn.evaluate --checkpoint <ckpt> --set evaluation.grid_n=301   # other export grid

# everything: 3 seeds + sensitivity run + summary (skips finished runs, resumes unfinished ones)
bash scripts/run_all.sh
python -m cavity_pinn.summarize      # -> runs/summary/
```

Any configuration value can be overridden with `--set section.key=value`.

## Layout

| Path | Purpose |
|---|---|
| `config/re100.yaml` | all settings: seeds, network, sampling, weights, optimisers, evaluation |
| `cavity_pinn/model.py` | MLP (2 → 6×64 tanh → 3), Xavier-normal weights, internal map [0,1]² → [-1,1]² |
| `cavity_pinn/physics.py` | autograd derivatives and NS residuals w.r.t. the physical coordinates; vorticity |
| `cavity_pinn/sampling.py` | Sobol + near-wall interior sampling, open-segment boundary sampling, corner distances |
| `cavity_pinn/train.py` | Adam → L-BFGS training, logging, resumable checkpoints |
| `cavity_pinn/benchmark.py` | Ghia CSV loader and error metrics (validation only) |
| `cavity_pinn/evaluate.py` | metrics, acceptance table, figures, CSV / Tecplot / VTK export |
| `cavity_pinn/summarize.py` | multi-seed table, variability, sensitivity comparison |
| `tests/` | verification tests (see below) |
| `data/` | Ghia CSVs, provenance and source snapshots |
| `runs/<run>/` | `config.yaml`, `run_info.json`, `train_history.csv`, `checkpoints/`, `eval/` |

## Formulation

The domain is [0,1]². Scales are the cavity width L, the lid speed U and the pressure ρU², with
Re = UL/ν = 100. The network outputs u, v and p. The residuals are:

```
r_c = u_x + v_y
r_u = u u_x + v u_y + p_x − (u_xx + u_yy)/Re
r_v = u v_x + v v_y + p_y − (v_xx + v_yy)/Re
```

These use nested reverse-mode autograd w.r.t. the physical (x, y). The [-1,1] input map is
inside `forward`, so the chain-rule factor 2 (or 4 for second derivatives) is included
automatically. Tests verify this.

Boundary conditions:

* Lid, y = 1 with 0 < x < 1: u = 1, v = 0. The lid speed is constant, not smoothed or tapered.
* Bottom, left and right walls: u = v = 0.
* Gauge: p(0.5, 0.5) = 0. No other pressure condition is imposed.

**Upper corners.** The prescribed velocity is discontinuous at (0,1) and (1,1). These two exact
points are excluded from boundary training: every wall is sampled on its open segment.
No finite exclusion zone is used. The nearest training point to each upper corner is logged in
`run_info.json` (`corner_report_boundary_training`). For seed 101 the distances are 7.3e-4 to the
upper-left corner and 2.4e-4 to the upper-right corner.

**Loss** (fixed weights, no adaptive weighting):

```
L = 1·⟨r_c²⟩ + 1·⟨r_u²⟩ + 1·⟨r_v²⟩ + 10·Σ_walls ⟨(u−u_b)² + (v−v_b)²⟩_wall + 1·p(0.5,0.5)²
```

Each ⟨·⟩_wall is a separate mean over that wall's 500 points.

## Sampling

The 10,000 interior points are a mixture of two parts:

* 80% scrambled Sobol points on (0,1)².
* 20% near-wall points. For each, a wall is picked uniformly, the normal distance is
  d ~ U(0, δ] with δ = 0.05, and the tangential position is uniform.

The resulting sampling density is

f(x,y) = (1 − α) + α/(4δ) · Σ_w 1[dist_w < δ],   with α = 0.2.

In a band f = 1.8; in a corner square where two bands overlap f = 2.8; elsewhere f = 0.8.
**The sample mean ⟨r²⟩ therefore estimates the weighted integral ∫ r² f dA**, which up-weights
the near-wall layers relative to a uniform-domain mean. `tests/test_sampling_boundary.py` checks
that f integrates to 1 and matches the empirical band fraction.

During Adam, the interior set is redrawn every 500 iterations from a seed derived
deterministically from the run seed and the refresh index. At the switch to L-BFGS, a dedicated
interior set is drawn and then frozen. The boundary points and the loss weights stay fixed for
the whole run.

## Optimisation

1. **Adam:** lr = 1e-3, decayed smoothly by 0.5 every 2000 steps, for 6000 iterations.
2. **L-BFGS:** PyTorch L-BFGS with a strong-Wolfe line search, history 50, for 4000 iterations.
   Each `step` call runs 50 iterations, and progress is logged and checkpointed between calls.
   The closure is deterministic, with fixed points and no randomness, and this is tested. If
   L-BFGS produces non-finite parameters, the step is reverted and the run stops with status
   `lbfgs_nonfinite_reverted`.

Logged quantities, every 50 Adam iterations and after every L-BFGS call, are in
`train_history.csv`:

* every unweighted loss component, plus the weights;
* the learning rate;
* optimizer loss and gradient evaluations (`n_evals`), plus logging-only evaluations (`n_diag_evals`);
* elapsed time;
* peak memory: process peak working set on CPU, or `max_memory_allocated` on CUDA.

Checkpoints (`checkpoint_last.pt`, `checkpoint_adam_end.pt`, `checkpoint_final.pt`) contain the
model, optimizer, scheduler, configuration, current collocation points, counters, history and
the torch/numpy/python RNG states.

The *final* checkpoint is always the one evaluated. No checkpoint or hyperparameter is selected
using Ghia errors.

## Evaluation (all on data independent of training)

* **Ghia profiles:** the network is evaluated *directly* at the 17 + 17 tabulated coordinates,
  with no interpolation. Reported: RMSE and max |error|.
* **PDE residuals:** RMS on 20,000 fixed uniform-random held-out interior points (seed 987654321).
  As a clearly labelled diagnostic only, the RMS excluding disks of radius 0.05 around the upper
  corners is also given.
* **Walls:** 4000 independent uniform-random points per open wall segment. Reported: RMS and
  max error of u and v, split into near-corner (within 0.05 of a wall end point) and rest.
* **Pressure gauge:** |p(0.5, 0.5)|.
* **Acceptance targets** (project targets, not Ghia's): centreline RMSE ≤ 0.01; centreline
  max |error| ≤ 0.03; each held-out PDE RMS ≤ 1e-3; each wall/component RMS ≤ 0.01;
  gauge ≤ 1e-4. Each is reported as PASS or FAIL.

### Outputs (`runs/<run>/eval/`)

* `metrics.json`
* `ghia_u_x0.5.csv`, `ghia_v_y0.5.csv`: reference values, predictions and errors
* `ghia_comparison.png`
* `fields_contours.png`: u, v, p, speed, vorticity ω = v_x − u_y (autograd), and streamlines
* `streamlines.png`
* `loss_history.png`
* `residual_maps.png`: log10|r| on the interior nodes of the 201² grid
* `fields_201x201.{csv,plt,vtk}`: the structured-field export

The structured export (configurable `evaluation.grid_n`, default 201×201, x varying fastest)
holds x, y, u, v, p and ω, plus an `upper_corner` flag.

**Corner convention in the exports:** every node holds the raw network output. The two exact
upper-corner nodes are flagged `upper_corner = 1`. They were never trained on and are excluded
from all quantitative diagnostics. Contour plots blank them, and residual maps use interior
nodes only.

## Verification tests (`pytest`)

* `test_derivatives.py`: first and second autograd derivatives through the input scaling vs
  analytic functions, for the unit box and a non-unit box; network derivatives vs finite
  differences; autograd vorticity; float64 and Xavier initialisation.
* `test_residuals.py`:
  * Kovasznay flow, an exact NS solution at Re = 100, gives zero residual;
  * a polynomial and a trigonometric manufactured field, with hand-derived residuals, run
    through the network wrapper;
  * evaluation-mode and training-mode residuals agree.
* `test_sampling_boundary.py`: boundary targets, open segments with upper corners excluded,
  corner-distance report, independent evaluation points, sampling density.
* `test_benchmark.py`: CSV loading, coordinate orientation, error formulas, evaluation at the
  exact tabulated points, and that the training modules never reference the benchmark.
* `test_checkpoint.py`:
  * save/load round trip;
  * resuming from mid-Adam and mid-L-BFGS snapshots reproduces the uninterrupted run bit for bit;
  * deterministic L-BFGS closure;
  * logged components recompose the total loss.
