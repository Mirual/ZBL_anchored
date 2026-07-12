#!/usr/bin/env bash
# Full DPA-3.1 anchor pipeline: rnd_build → tune → predict×3 → compare.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${ZBL_PYTHON:-python}
export OMP_NUM_THREADS=4

echo "[$(date +%H:%M:%S)] === rnd_build ==="
$PY scripts/rnd_build.py
echo "[$(date +%H:%M:%S)] === tune_pairphys ==="
$PY scripts/tune_pairphys.py
for cond in keep_test u200_test mptrj; do
  echo "[$(date +%H:%M:%S)] === predict $cond ==="
  $PY scripts/predict.py --cond "$cond"
done
echo "[$(date +%H:%M:%S)] === compare ==="
$PY scripts/compare.py
echo "[$(date +%H:%M:%S)] === DONE ==="
