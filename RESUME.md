# Resuming the Re = 100 cavity PINN experiments

State when stopped (2026-10-11 18:39):

* Seed 101 was in the Adam phase. Its last checkpoint is at Adam iteration 1500 of 6000, after
  1866 s of training: `runs/re100_seed101/checkpoints/checkpoint_last.pt`.
* Seeds 202 and 303 and the sensitivity run (width 96, 20k interior points, seed 101) have not
  started.
* Code, the 21 passing tests, the smoke run (`runs/re100_smoke_seed101`), the Ghia CSVs with
  provenance, and the README are complete.
* Still to do: `VALIDATION_REPORT.md`.

## Command only

```bash
cd D:/5_PINNs_HMT/satya_pinns_HMT
bash scripts/run_all.sh > runs/run_all_resume.log 2>&1
```

The script resumes seed 101 from `checkpoint_last.pt`. It then trains seeds 202 and 303 and the
sensitivity run, evaluates each one, and writes `runs/summary/`. Finished runs are skipped.
About 26 h of CPU time remain on the i3-1115G4.

## Prompt for a new Claude Code session

> Continue the Re = 100 lid-driven cavity PINN project in D:\5_PINNs_HMT\satya_pinns_HMT. Read
> README.md, RESUME.md and data/PROVENANCE.md first. The implementation and tests are complete,
> so do not rewrite them. Run `.venv/Scripts/python -m pytest -q` to confirm all 21 tests still
> pass.
>
> Then resume the experiment queue in the background with
> `bash scripts/run_all.sh > runs/run_all_resume.log 2>&1`. Seed 101 resumes from
> runs/re100_seed101/checkpoints/checkpoint_last.pt at Adam iteration 1500. The queue then runs
> seeds 202 and 303 and the predeclared sensitivity run (seed 101, model.width=96,
> sampling.n_interior=20000). Check that no other training process is running before you start.
>
> When seed 101 reaches L-BFGS and when it finishes, inspect its training history for stalls,
> non-finite values or early L-BFGS termination. Diagnose any failure before changing a budget,
> and document any change.
>
> When all runs are evaluated, write VALIDATION_REPORT.md. It must include:
> - the actual hardware, configurations, seeds and training times;
> - data provenance: the Ghia transcription is UNVERIFIED against the original and cross-checked
>   against two secondary sources;
> - every acceptance criterion as PASS or FAIL for all three seeds, with their variability and
>   baseline seed 101 reported;
> - the sensitivity results;
> - near-corner errors;
> - a clear split between implemented functionality, completed runs and remaining work.
>
> Never use Ghia errors for model, checkpoint or hyperparameter selection. Never relax thresholds
> or fabricate metrics. Finish by listing changed files, tests executed, completed experiments,
> measured results and unresolved limitations.
