# Project status: Re = 100 lid-driven cavity PINN

*Last updated: 2026-10-11 21:55*

## Summary

**The seed 101 run is complete and evaluated. It does not meet the acceptance targets: 7 PASS,
9 FAIL.** Full results are in [VALIDATION_REPORT.md](VALIDATION_REPORT.md). Following your
instruction, only seed 101 was run.

## Seed 101 results (final checkpoint; no selection)

| Criterion (target) | Measured | Result |
|---|---|---|
| Centreline u RMSE (≤ 0.01) | 0.0320 | FAIL |
| Centreline u max \|err\| (≤ 0.03) | 0.0634 | FAIL |
| Centreline v RMSE (≤ 0.01) | 0.0340 | FAIL |
| Centreline v max \|err\| (≤ 0.03) | 0.0565 | FAIL |
| Held-out continuity RMS (≤ 1e-3) | 0.109 | FAIL |
| Held-out x-momentum RMS (≤ 1e-3) | 0.958 | FAIL |
| Held-out y-momentum RMS (≤ 1e-3) | 0.417 | FAIL |
| Lid u / v RMS (≤ 0.01) | 0.0095 / 0.0035 | PASS / PASS |
| Bottom u / v RMS (≤ 0.01) | 0.0015 / 0.0019 | PASS / PASS |
| Left u / v RMS (≤ 0.01) | 0.0253 / 0.0022 | FAIL / PASS |
| Right u / v RMS (≤ 0.01) | 0.0153 / 0.0050 | FAIL / PASS |
| Pressure gauge (≤ 1e-4) | 6.0e-5 | PASS |

These targets are project targets, not criteria from Ghia et al. The Ghia transcription is
unverified against the original paper; see `data/PROVENANCE.md`.

## Run facts

* **Hardware:** Intel i3-1115G4 CPU with 4 threads and no GPU. Peak memory was 1.27 GB.
* **Training:** Adam for 6000 iterations, then L-BFGS for 4000; 10,599 optimizer evaluations in
  3.11 h. The run was stopped and resumed once, bit-exactly.
* **Stopping reason:** the iteration budget ran out. The training loss was still falling (39%
  over the last 2000 iterations), so **training had not converged.**

## Main findings

* **Flow pattern:** qualitatively correct, with one clockwise primary vortex. The centreline
  extrema are about 20–25% too weak.
* **Under-fitting:** away from the singular upper corners, the held-out PDE residual RMS is
  0.02–0.03. That is about 20–30× the target and matches the training residual. So the network
  is under-fitted everywhere, not over-fitted.
* **Corner residuals:** inside r < 0.05 of the two upper corners, residuals are huge (RMS 2–18),
  where the lid speed jumps. These dominate the uniform held-out RMS.
* **Corner wall errors:** the side-wall u failures are confined to within 0.05 of the upper
  corners. Elsewhere, every wall error is below 0.004.

## Not done

* Seeds 202 and 303 (seed variability) and the sensitivity run were not run, because you
  restricted the scope to seed 101.
* No remediation runs were made.

## Options (your decision)

1. **Extend L-BFGS**, justified because the training loss had not converged. This takes about
   1 s per iteration:
   `.venv/Scripts/python -m cavity_pinn.train --resume runs/re100_seed101/checkpoints/checkpoint_final.pt --set lbfgs.iterations=10000`
2. **Run the remaining seeds and the sensitivity run:** `bash scripts/run_all.sh`. This takes
   about 23 h.
3. **Make documented method changes:** corner-refined or residual-based sampling, a larger
   network, or a treatment of the corner singularity.

Even with longer training, the uniform 1e-3 PDE target is probably unattainable while the
singular lid is kept with no corner exclusion zone.
