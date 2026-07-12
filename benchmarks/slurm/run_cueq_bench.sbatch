#!/usr/bin/env bash
#SBATCH --job-name=mace_cueq_bench
#SBATCH --partition=${ZBL_SLURM_PARTITION:-main}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# cuEquivariance speedup-vs-size sweep on Blackwell (RTX PRO 6000, sm_120).
# We run MACE-MH-1 single-point OFF (e3nn) vs ON (fused cueq) on large
# cells, in fp64 and fp32, and compare: lossless + per-structure speedup.
# Requires an env with working fused cueq (cuequivariance-ops-torch-cu12) — mace-cueq.

set -euo pipefail

CONDA_ENV="${ZBL_CONDA_ENV:-/path/to/conda-env}"   # TODO: set for your machine (env with working fused cueq)
PYTHON="${CONDA_ENV}/bin/python"
export PATH="${CONDA_ENV}/bin:${PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1          # MACE-MH-1 only loads this way

BENCH="${ZBL_BENCH_DIR:-/path/to/bench}"   # TODO: set for your machine (mace_mlip_work/bench)
RUNS="${BENCH}/runs"
MODEL="${ZBL_MACE_MH1:-/path/to/mace-mh-1.model}"
STRUCT="${RUNS}/structures_large.extxyz"
HEAD="omat_pbe"
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"

mkdir -p "${RUNS}/logs"
echo "host:    $(hostname)"
echo "job_id:  ${SLURM_JOB_ID:-?}"
echo "gpu:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "env:     ${CONDA_ENV}"
echo "model:   ${MODEL}"
echo "started: $(date -Is)"
"${PYTHON}" -c 'import torch,cuequivariance_torch as c; print("cuequivariance_torch",c.__version__,"| torch",torch.__version__)'
echo

# structures are built once and reused byte-for-byte across all runs
[ -f "${STRUCT}" ] || "${PYTHON}" "${BENCH}/gen_structures_large.py" "${STRUCT}"

run () {  # $1=dtype  $2=suffix(off|on)  $3...=extra flags (e.g. --enable-cueq)
    local dt="$1"; local sfx="$2"; shift 2
    echo "### dtype=${dt} ${sfx} $*"
    "${PYTHON}" "${BENCH}/bench_mace.py" --model "${MODEL}" --device cuda \
        --dtype "${dt}" --head "${HEAD}" --structures "${STRUCT}" \
        --repeats "${REPEATS}" --warmup "${WARMUP}" \
        --out "${RUNS}/cueq_${dt}_${sfx}.json" --label "${dt}_${sfx}" "$@"
}

echo "### [1/4] fp64 e3nn (OFF)";       run float64 off
echo; echo "### [2/4] fp64 cueq (ON)";  run float64 on  --enable-cueq
echo; echo "### [3/4] fp32 e3nn (OFF)"; run float32 off
echo; echo "### [4/4] fp32 cueq (ON)";  run float32 on  --enable-cueq

cmp () {  # $1=ref  $2=test  $3...=tol flags  (a DIFF does not fail the job)
    set +e
    "${PYTHON}" "${BENCH}/compare_runs.py" "$1" "$2" "${@:3}"
    echo "[compare exit=$?]"
    set -e
}

echo; echo "===== A: fp64 OFF vs ON  (cueq lossless + speedup vs size) ====="
cmp "${RUNS}/cueq_float64_off.json" "${RUNS}/cueq_float64_on.json" --etol 1e-5 --ftol 1e-4 --stol 1e-5
echo; echo "===== B: fp32 OFF vs ON  (cueq speedup vs size, fp32) ====="
cmp "${RUNS}/cueq_float32_off.json" "${RUNS}/cueq_float32_on.json" --etol 2e-3 --ftol 2e-2 --stol 1e-3
echo; echo "===== C: fp64 OFF vs fp32 ON  (combined precision+cueq) ====="
cmp "${RUNS}/cueq_float64_off.json" "${RUNS}/cueq_float32_on.json" --etol 5e-3 --ftol 5e-2 --stol 1e-3

echo; echo "finished: $(date -Is)"
