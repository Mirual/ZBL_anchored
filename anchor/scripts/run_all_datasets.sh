#!/usr/bin/env bash
set -uo pipefail; export PATH="${ZBL_CONDA_BIN:-/path/to/conda/env/bin}:$PATH"  # TODO: set for your machine
cd "$(dirname "$0")/.."
PY="${ZBL_PYTHON:-python}"
CFG="--A 800 --b 0.5 --rho-lo 3.0 --rho-hi 4.0 --power 2"
for d in "u200:${ZBL_EVAL_DATA:-/path/to/vasp_eval/preflight}/splits/u200_test.xyz" "keep:${ZBL_EVAL_DATA:-/path/to/vasp_eval/preflight}/splits/keep_test.xyz" "keepfull:${ZBL_EVAL_DATA:-/path/to/vasp_eval/preflight}/splits/compression_kept.xyz" "mptrj:${ZBL_MPTRJ_XYZ:-/path/to/mptrj_stratified_10k.xyz}"; do
  name=${d%%:*}; data=${d#*:}
  echo "### anchor $name $(date -Is)"
  $PY scripts/rho_anchor_predict.py --data "$data" --out results/all_anchor_$name.json $CFG 2>&1 | grep frames | tail -1
done
echo "### DONE $(date -Is)"
