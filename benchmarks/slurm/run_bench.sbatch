#!/usr/bin/env bash
#SBATCH --job-name=mace_bench
#SBATCH --partition=${ZBL_SLURM_PARTITION:-main}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

CONDA_ENV="${ZBL_CONDA_ENV:-/path/to/conda-env}"   # TODO: set for your machine (conda env with MACE 0.3.15)
PYTHON="${CONDA_ENV}/bin/python"
export PATH="${CONDA_ENV}/bin:${PATH}"

BENCH="${ZBL_BENCH_DIR:-/path/to/bench}"   # TODO: set for your machine (mace_mlip_work/bench)
RUNS="${BENCH}/runs"
MODEL="${ZBL_MACE_MH1:-/path/to/mace-mh-1.model}"
STRUCT="${RUNS}/structures.extxyz"      # built once, reused byte-identical across runs

echo "host:    $(hostname)"
echo "job_id:  ${SLURM_JOB_ID:-?}"
echo "gpu:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "env:     ${CONDA_ENV}"
echo "model:   ${MODEL}"
echo "started: $(date -Is)"
echo

# --- Run 1: fp64 baseline (the reference, analogous to stock.json) ----------
echo "### [1/3] fp64 baseline"
"${PYTHON}" "${BENCH}/bench_mace.py" --model "${MODEL}" --device cuda \
    --dtype float64 --head omat_pbe --structures "${STRUCT}" \
    --out "${RUNS}/fp64.json" --label fp64

# --- Run 2: fp64 again (determinism / parity-path check) --------------------
echo; echo "### [2/3] fp64 repeat (determinism check)"
"${PYTHON}" "${BENCH}/bench_mace.py" --model "${MODEL}" --device cuda \
    --dtype float64 --head omat_pbe --structures "${STRUCT}" \
    --out "${RUNS}/fp64_b.json" --label fp64_b

# --- Run 3: fp32 speed knob (audit lever: ~2x bandwidth, sub-meV) -----------
echo; echo "### [3/3] fp32 speed knob"
"${PYTHON}" "${BENCH}/bench_mace.py" --model "${MODEL}" --device cuda \
    --dtype float32 --head omat_pbe --structures "${STRUCT}" \
    --out "${RUNS}/fp32.json" --label fp32

# --- Comparisons (don't let a DIFF kill the job; both are informative) ------
echo; echo "================ COMPARE A: fp64 vs fp64 (tight tol -> expect PASS) ================"
set +e
"${PYTHON}" "${BENCH}/compare_runs.py" "${RUNS}/fp64.json" "${RUNS}/fp64_b.json"
echo "[compare A exit=$?]"

echo; echo "================ COMPARE B: fp64 vs fp32 (relaxed tol -> PASS + speedup) ==========="
"${PYTHON}" "${BENCH}/compare_runs.py" "${RUNS}/fp64.json" "${RUNS}/fp32.json" \
    --etol 5e-3 --ftol 2e-2 --stol 1e-4
echo "[compare B exit=$?]"
set -e

echo; echo "finished: $(date -Is)"
