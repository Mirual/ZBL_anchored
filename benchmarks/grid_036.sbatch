#!/usr/bin/env bash
#SBATCH --job-name=grid036
#SBATCH --partition=${ZBL_SLURM_PARTITION:-main}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# Full uniform grid (config x size) through bench_compile_036.py, all rows one
# methodology (same warmup/timing/peak-mem). Runs INSIDE one sbatch allocation
# so a Claude Code restart can't kill it. Resumable: skips any (combo,natoms,
# dtype) already present in the RESULT log.
# ---------------------------------------------------------------------------
set -u
PY="${ZBL_PYTHON:-python}"
cd "${ZBL_BENCH_DIR:-/path/to/bench}"   # TODO: set for your machine (mace_mlip_work/bench)
LOG=runs/logs/grid_036.out
touch "$LOG"

CONFIGS=("e3nn float64" "e3nn float32" "cueq float64" "cueq float32" \
         "cmp-e3nn float32" "cmp+cueq float32")
NNS=(3 5 7 8 9 10)   # Cu fcc primitive: 27 125 343 512 729 1000 atoms

for nn in "${NNS[@]}"; do
  natoms=$(( nn * nn * nn ))
  for cfg in "${CONFIGS[@]}"; do
    set -- $cfg; combo=$1; dtype=$2
    if grep -q "\"combo\": \"$combo\", \"natoms\": $natoms, \"dtype\": \"$dtype\"" "$LOG"; then
      echo "### skip (done) nn=$nn combo=$combo dtype=$dtype"
      continue
    fi
    echo "### run nn=$nn combo=$combo dtype=$dtype"
    TORCHINDUCTOR_COMPILE_THREADS=1 $PY -u bench_compile_036.py \
        --combo "$combo" --nn "$nn" --dtype "$dtype" 2>/dev/null \
        | grep "RESULT_JSON" >> "$LOG"
  done
done
echo "GRID DONE" >> "$LOG"
