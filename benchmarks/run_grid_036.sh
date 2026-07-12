#!/usr/bin/env bash
# Full uniform grid: every (config x size) through bench_compile_036.py so all
# rows share one timing/memory methodology. One srun per cell (compile is
# RAM-heavy + recompiles per shape -> isolate). Appends RESULT_JSON to LOG.
set -u
PY="${ZBL_PYTHON:-python}"
cd "${ZBL_BENCH_DIR:-/path/to/bench}"   # TODO: set for your machine (mace_mlip_work/bench)
LOG=runs/logs/grid_036.out
: > "$LOG"

# config: "combo dtype"
CONFIGS=("e3nn float64" "e3nn float32" "cueq float64" "cueq float32" \
         "cmp-e3nn float32" "cmp+cueq float32")
NNS=(3 5 7 8 9 10)   # Cu fcc primitive: 27 125 343 512 729 1000 atoms

for nn in "${NNS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    set -- $cfg; combo=$1; dtype=$2
    echo "### nn=$nn combo=$combo dtype=$dtype" | tee -a "$LOG"
    srun --partition="${ZBL_SLURM_PARTITION:-main}" --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=00:30:00 \
         --job-name="g${nn}_${combo}" \
         bash -c "TORCHINDUCTOR_COMPILE_THREADS=1 $PY -u bench_compile_036.py \
                  --combo $combo --nn $nn --dtype $dtype" \
         2>&1 | grep -E "RESULT_JSON|Killed|CANCELLED|out of memory|Error" | tee -a "$LOG"
  done
done
echo "GRID DONE" | tee -a "$LOG"
