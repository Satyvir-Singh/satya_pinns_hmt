#!/usr/bin/env bash
# Full experiment queue: three predefined seeds, then the predeclared sensitivity run,
# then evaluation and the multi-seed summary.  Re-running this script is safe:
# finished runs are skipped and interrupted runs resume from checkpoint_last.pt.
#
#   bash scripts/run_all.sh            (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
CFG=config/re100.yaml
SENS_SET="model.width=96 sampling.n_interior=20000"

train_or_resume () {   # $1 = run dir, rest = args for a fresh start
  local dir=$1; shift
  if [ -f "$dir/checkpoints/checkpoint_final.pt" ]; then
    echo "== $dir already finished"
  elif [ -f "$dir/checkpoints/checkpoint_last.pt" ]; then
    echo "== resuming $dir"; "$PY" -m cavity_pinn.train --resume "$dir/checkpoints/checkpoint_last.pt"
  else
    echo "== training $dir"; "$PY" -m cavity_pinn.train --config "$CFG" "$@"
  fi
  "$PY" -m cavity_pinn.evaluate --checkpoint "$dir/checkpoints/checkpoint_final.pt"
}

for seed in 101 202 303; do
  train_or_resume "runs/re100_seed$seed" --seed "$seed"
done
train_or_resume runs/re100_sens_w96_n20k_seed101 --seed 101 --tag _sens_w96_n20k --set $SENS_SET

"$PY" -m cavity_pinn.summarize
