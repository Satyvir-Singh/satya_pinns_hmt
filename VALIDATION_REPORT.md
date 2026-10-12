# Validation report: PINN for the steady lid-driven cavity at Re = 100

*Date: 2026-10-11. Scope: seed 101 only, by instruction. Seeds 202 and 303 and the sensitivity
run were not executed.*

## 1. Outcome in one paragraph

The seed 101 baseline (Adam for 6000 iterations, then L-BFGS for 4000; 3.1 h on a laptop CPU)
recovers the correct flow pattern. That is one clockwise primary vortex with its centre near
(0.62, 0.76), read approximately from the streamline plot, with the expected velocity and vorticity structure. **However, it fails 9 of the 16
predeclared acceptance criteria.** Specifically:

* Both Ghia centreline profiles have RMSE of about 0.032–0.034 against a target of 0.01. The
  peak velocities are under-predicted by roughly 20%.
* All three held-out PDE residual RMS values are 0.11, 0.96 and 0.42 against a target of 1e-3.
  Outside small disks around the singular upper corners they are 0.019–0.029, which is still
  about 20–30× the target.
* The u velocity on the left and right walls fails, with all of the error concentrated within
  0.05 of the upper corners.

The bottom and top walls, both v-components on the side walls and the pressure gauge pass.
**This run is not a validated solution.** The training loss was still decreasing when the
iteration budget ran out.

## 2. Setup actually used

| Item | Value |
|---|---|
| Hardware | Intel Core i3-1115G4 (2 cores / 4 threads, 3.0 GHz), 15.7 GB RAM, **no GPU**; Windows 11 Enterprise |
| Software | Python 3.13.15, PyTorch 2.14.1+cpu (4 intra-op threads), NumPy 2.5.4, SciPy 1.18.1 |
| Device selected | `cpu` (auto-detected: `torch.cuda.is_available() = False`) |
| Precision | float64 throughout |
| Network | 2 → 6 hidden layers × 64 (tanh) → 3 linear outputs; Xavier-normal weights, zero biases; inputs mapped to [-1,1]² inside the model |
| Interior points | 10,000: 8000 scrambled Sobol plus 2000 near-wall (δ = 0.05). Redrawn every 500 Adam iterations; a dedicated frozen set for L-BFGS |
| Boundary points | 500 per wall on the open segments; upper corners excluded |
| Nearest boundary training point to the upper corners | upper-left 7.28e-4, upper-right 2.40e-4 (no finite exclusion zone) |
| Loss weights (fixed) | continuity 1, x-momentum 1, y-momentum 1, sum of 4 per-wall mean losses 10, gauge 1 |
| Adam | 6000 iterations; lr 1e-3 × 0.5^(it/2000), final lr 1.25e-4 |
| L-BFGS | 4000 iterations; strong-Wolfe line search; history 50; 50 iterations per call |
| Optimizer loss evaluations | 10,599 (6000 Adam + 4599 L-BFGS) |
| Training time | 11,186 s = 3.11 h, plus about 7 min of discarded work from a requested stop/resume at iteration 1500 (bit-exact resume) |
| Peak memory | 1.27 GB (process peak working set) |
| Stopping | budget exhausted (`status = done`), **not** the L-BFGS tolerance |
| Seeds | predefined 101, 202, 303; baseline 101 chosen in advance; **only 101 was run** |
| Checkpoint evaluated | the final one, with no selection of any kind |

Configuration: `config/re100.yaml`. The copy actually used is at `runs/re100_seed101/config.yaml`.

**Sampling note.** The interior sample mean estimates ∫ r² f dA with
f = 0.8 + 0.2/(4δ) Σ_w 1[dist_w < δ]. That is a weighted residual objective, weighting the
near-wall bands 1.8× and the corners 2.8× relative to the interior's 0.8. It is not a uniform
domain mean.

No adaptive weighting, residual-based sampling or architecture change was used.

## 3. Training history (unweighted training-set mean squares)

| Stage | Total (weighted) | Continuity | x-momentum | y-momentum | Lid | Bottom | Left | Right | Gauge |
|---|---|---|---|---|---|---|---|---|---|
| Adam it. 1 | 9.14 | 1.4e-1 | 1.7e-1 | 5.0e-2 | 6.4e-1 | 1.4e-1 | 5.3e-2 | 5.3e-2 | 0 |
| Adam 1000 | 3.41e-1 | 1.1e-2 | 3.2e-2 | 1.6e-2 | 6.7e-3 | 2.2e-5 | 8.4e-3 | 1.3e-2 | 2e-7 |
| Adam 6000 | 5.45e-2 | 2.3e-3 | 2.3e-3 | 2.3e-3 | 1.4e-3 | 2.7e-6 | 1.6e-3 | 1.7e-3 | 1e-7 |
| L-BFGS 0 (new frozen set) | 6.14e-2 | 1.5e-3 | 8.3e-3 | 3.8e-3 | 1.5e-3 | 3.0e-6 | 1.5e-3 | 1.7e-3 | 2e-8 |
| L-BFGS 2000 | 1.07e-2 | 7.1e-4 | 1.1e-3 | 1.0e-3 | 2.6e-4 | 3.8e-6 | 4.1e-5 | 4.8e-4 | 2e-9 |
| L-BFGS 4000 (final) | 6.52e-3 | 5.2e-4 | 8.1e-4 | 8.9e-4 | 1.4e-4 | 6.0e-6 | 8.7e-6 | 2.8e-4 | 4e-9 |

Over the last 2000 L-BFGS iterations the loss fell by 39%, so the optimisation had **not
converged**. On the first frozen L-BFGS set, the x-momentum term jumped 3.6×, from 2.3e-3 to
8.3e-3. This indicates sample-dependent, localised high-residual regions near the singular
corners; L-BFGS absorbed the jump within 50 iterations.

Figure: `runs/re100_seed101/eval/loss_history.png`.

## 4. Validation results and acceptance criteria

These are project targets, not criteria from Ghia et al. Thresholds were not changed.

| # | Criterion | Target | Seed 101 | Result |
|---|---|---|---|---|
| 1 | Centreline u RMSE (Table I, 17 pts) | ≤ 0.01 | 3.195e-2 | **FAIL** |
| 2 | Centreline u max \|err\| | ≤ 0.03 | 6.340e-2 | **FAIL** |
| 3 | Centreline v RMSE (Table II, 17 pts) | ≤ 0.01 | 3.397e-2 | **FAIL** |
| 4 | Centreline v max \|err\| | ≤ 0.03 | 5.654e-2 | **FAIL** |
| 5 | Held-out continuity RMS (20,000 uniform pts) | ≤ 1e-3 | 1.087e-1 | **FAIL** |
| 6 | Held-out x-momentum RMS | ≤ 1e-3 | 9.577e-1 | **FAIL** |
| 7 | Held-out y-momentum RMS | ≤ 1e-3 | 4.167e-1 | **FAIL** |
| 8 | Lid u RMS | ≤ 0.01 | 9.541e-3 | PASS |
| 9 | Lid v RMS | ≤ 0.01 | 3.462e-3 | PASS |
| 10 | Bottom u RMS | ≤ 0.01 | 1.465e-3 | PASS |
| 11 | Bottom v RMS | ≤ 0.01 | 1.918e-3 | PASS |
| 12 | Left u RMS | ≤ 0.01 | 2.529e-2 | **FAIL** |
| 13 | Left v RMS | ≤ 0.01 | 2.162e-3 | PASS |
| 14 | Right u RMS | ≤ 0.01 | 1.527e-2 | **FAIL** |
| 15 | Right v RMS | ≤ 0.01 | 5.036e-3 | PASS |
| 16 | Pressure gauge \|p(0.5,0.5)\| | ≤ 1e-4 | 5.97e-5 | PASS |

**Result: 7 PASS, 9 FAIL.**

### 4.1 Ghia centreline comparison
The network was evaluated directly at the tabulated coordinates, with no interpolation. The
files are:

* `eval/ghia_u_x0.5.csv` and `eval/ghia_v_y0.5.csv`, which hold the reference values,
  predictions and errors;
* `eval/ghia_comparison.png`.

| Location | Ghia | PINN | Error |
|---|---|---|---|
| u at y = 0.4531 (min of u) | −0.21090 | −0.1615 | +0.049 |
| u at y = 0.8516 | 0.23151 | 0.1681 | −0.063 (max \|err\|) |
| v at x = 0.2344 (max of v) | 0.17527 | 0.1342 | −0.041 |
| v at x = 0.8047 (min of v) | −0.24533 | −0.1888 | +0.057 (max \|err\|) |

The profile shapes and zero crossings agree. The extrema are about 20–25% too weak, which looks
like an under-resolved, over-damped solution. Errors at the wall points y = 0, y = 1, x = 0 and
x = 1 are ≤ 0.002.

### 4.2 Held-out PDE residuals (20,000 uniform points, seed 987654321)

| Residual | RMS (criterion) | max \|r\| | RMS outside r < 0.05 of upper corners* | RMS inside those disks* |
|---|---|---|---|---|
| Continuity | 1.09e-1 | 15.0 | 1.95e-2 | 1.99 |
| x-momentum | 9.58e-1 | 133.7 | 2.86e-2 | 17.8 |
| y-momentum | 4.17e-1 | 56.9 | 2.92e-2 | 7.72 |

\*These two columns are diagnostic only and are not the acceptance criterion. 58 of the 20,000
points fall inside the corner disks, and the nearest held-out point is 0.011 from an upper corner.

Away from the corners, the held-out RMS (0.02–0.03) matches the training-set RMS
(√8e-4 ≈ 0.028). So the network **generalises consistently, but is under-fitted everywhere**, not
only near the corners. The residual maps (`eval/residual_maps.png`) show |r| of about 1e-2 to
1e-1 over the whole cavity, with the maxima at the two upper corners.

### 4.3 Boundary errors (4000 independent uniform points per open wall)
"Near-corner" means within 0.05 of either end point of the wall. For the lid and side walls that
includes a singular upper corner.

| Wall | Comp. | RMS | Near-corner RMS | Near-corner max | Rest RMS | Rest max |
|---|---|---|---|---|---|---|
| Lid | u | 9.54e-3 | 3.10e-2 | 0.219 | 1.94e-3 | 4.29e-3 |
| Lid | v | 3.46e-3 | 9.67e-3 | 0.075 | 1.96e-3 | 4.84e-3 |
| Bottom | u | 1.47e-3 | 2.04e-3 | 2.42e-3 | 1.39e-3 | 2.37e-3 |
| Bottom | v | 1.92e-3 | 1.62e-3 | 3.02e-3 | 1.95e-3 | 3.95e-3 |
| Left | u | 2.53e-2 | 7.46e-2 | **0.900** | 9.1e-4 | 3.9e-3 |
| Left | v | 2.16e-3 | 5.43e-3 | 0.040 | 1.21e-3 | 1.95e-3 |
| Right | u | 1.53e-2 | 4.70e-2 | **0.630** | 7.2e-4 | 1.8e-3 |
| Right | v | 5.04e-3 | 1.45e-2 | 0.132 | 1.90e-3 | 3.61e-3 |

Away from the corners every wall is satisfied to better than 0.004 (max \|err\|). The two
side-wall u failures come entirely from the network smoothing the lid's u = 1 into the side
walls just below the upper corners.

### 4.4 Pressure gauge
p(0.5, 0.5) = −5.97e-5 (target ≤ 1e-4): **PASS**.

## 5. Diagnosis of the failures (no Ghia data used)

1. **Optimisation was not converged.** The training loss was still falling 39% per 2000 L-BFGS
   iterations at the end. The run stopped on its iteration budget, not on a tolerance.
2. **Under-fitting, not over-fitting.** The held-out residual away from the corners matches the
   training residual, so there is no generalisation gap. The network has not yet driven the
   residual down anywhere.
3. **Corner singularity.** The constant-speed lid makes velocity discontinuous at (0,1) and
   (1,1). The true solution then has unbounded vorticity and pressure there. Residuals inside the
   0.05 disks are 2–18 RMS, which dominates the uniform-domain criterion. The required set-up
   keeps the singular lid and uses uniform held-out points with no exclusion zone. Under those
   conditions, the 1e-3 RMS PDE target is likely unattainable for a smooth tanh network, even with
   much more training. This is an assessment, not something tested here. The same singularity
   produces the side-wall u failures.
4. **The centreline errors** are plausibly a result of points 1 and 3: a globally under-resolved
   and over-damped solution, and singular-corner errors competing for capacity in the loss. That
   explanation has not been demonstrated.

## 6. Data provenance

The Ghia et al. (1982) Re = 100 data come from Table I (u at x = 0.5) and Table II (v at y = 0.5),
in `data/*.csv`. **The transcription is UNVERIFIED against the original article**, because the
publisher page returned HTTP 403. All 34 coordinates and 34 values agree exactly with two
independent secondary sources:

* a GitHub gist by "ivan-pi" (2019);
* an Uppsala University degree project (2022, DiVA diva2:1668016).

Some circulating copies swap the v-values at x = 0.9531 and 0.9453. See `data/PROVENANCE.md`.

The Ghia data were used only for this post-training evaluation. They were never used in
training, sampling, weighting, stopping or checkpoint selection, and a test enforces this.
Ghia's values are a numerical 129×129 solution, not an exact one.

## 7. Implemented vs. completed vs. remaining

**Implemented and tested** (21 tests pass):

* autograd first and second derivatives through the input scaling;
* manufactured and Kovasznay residual checks;
* open-segment boundary sampling with upper-corner exclusion and distance logging;
* the sampling density;
* the Ghia loader, orientation checks and error formulas;
* checkpoint round trip and bit-exact resume across Adam and L-BFGS;
* a deterministic L-BFGS closure;
* the evaluation, figures, CSV, Tecplot and VTK exports, and the multi-seed summary script.

**Completed runs:**

* the smoke run `runs/re100_smoke_seed101`, which checks the pipeline only and is not a result;
* the **full baseline run for seed 101**, `runs/re100_seed101`.

**Outputs for seed 101** (`runs/re100_seed101/eval/`):

* `metrics.json`
* Ghia CSVs and plot
* `fields_contours.png` (u, v, p, speed, vorticity)
* `streamlines.png`
* `loss_history.png`
* `residual_maps.png`
* `fields_201x201.csv`, `.plt` and `.vtk`, where the exact upper corners are flagged
  `upper_corner=1`

**Not done:**

* Seeds 202 and 303, so there is **no seed-variability estimate**.
* The sensitivity study (seed 101, width 96, 20,000 interior points), so there is **no
  sensitivity result**.
* Any convergence or remediation runs.

## 8. Possible next steps (each to be justified without Ghia data)

* **Extend L-BFGS**, which is justified because the training loss had not converged:
  `python -m cavity_pinn.train --resume runs/re100_seed101/checkpoints/checkpoint_final.pt --set lbfgs.iterations=10000`
  Judge the result by training loss and held-out residuals. About 1 s per iteration on this CPU.
* Run seeds 202 and 303 and the sensitivity study: `bash scripts/run_all.sh`. About 23 h on this
  CPU.
* Methodological changes that would have to be documented as deviations:
  * residual-based or corner-refined sampling;
  * a larger network;
  * a reformulation that handles the corner singularity, for example a singular-solution
    subtraction.
  Smoothing or tapering the lid is excluded by the specification.
