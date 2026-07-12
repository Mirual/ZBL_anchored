#!/usr/bin/env bash
# ============================================================================
# setup_cueq_env.sh
#
# Creates a separate conda environment with cuEquivariance fused kernels for MACE
# by CLONING the known-working `gfnff-delta-mace` (torch 2.11+cu128, mace 0.3.15,
# e3nn 0.4.4) — torch+CUDA there already provably compute on Blackwell sm_120,
# so all that is left is to deliver cuequivariance-torch.
#
# Then — verification on GPU via srun: single-point MACE-MH-1 with enable_cueq
# OFF vs ON, checking losslessness (energies/forces match) and measuring speedup.
#
# Run:   bash setup_cueq_env.sh
# Options (env vars):
#   NEW_ENV=/path        where to put the new env (default below)
#   CUEQ_SPEC="cuequivariance-torch==0.X.Y"   version pin (default: latest)
#   FORCE=1              recreate env if it already exists
#   SKIP_VERIFY=1        do not run GPU verification
# ============================================================================
set -euo pipefail

# ---- config ----------------------------------------------------------------
SRC_ENV="${SRC_ENV:-/path/to/conda-env}"   # TODO: set for your machine (Blackwell-ready donor base)
NEW_ENV="${NEW_ENV:-/path/to/conda-env}"   # TODO: set for your machine (target env)
CUEQ_SPEC="${CUEQ_SPEC:-cuequivariance-torch}"                    # latest; pin -> "...==0.X.Y"
NV_INDEX="https://pypi.nvidia.com"                                # extra index for cueq-ops
MODEL="${MODEL:-${ZBL_MACE_MH1:-/path/to/mace-mh-1.model}}"
HEAD="${HEAD:-omat_pbe}"
CONDA_BASE="${CONDA_BASE:-/path/to/anaconda3}"   # TODO: set for your machine (conda base install)

echo "==> source env : ${SRC_ENV}"
echo "==> new env     : ${NEW_ENV}"
echo "==> cueq spec   : ${CUEQ_SPEC}"
echo "==> model       : ${MODEL}"
echo

# ---- 0. sanity -------------------------------------------------------------
[ -x "${SRC_ENV}/bin/python" ] || { echo "ERROR: source env not found: ${SRC_ENV}"; exit 1; }

# load conda into the shell
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found in ${CONDA_BASE}/etc/profile.d/ — set CONDA_BASE=..."; exit 1
fi

# ---- 1. clone --------------------------------------------------------------
# We consider an env valid only if bin/python exists; a broken partial (from an
# interrupted clone — e.g. a network IncompleteRead) is detected and removed
# automatically.
env_ok() { [ -x "${1}/bin/python" ]; }
nuke_env() { conda env remove -p "${1}" --yes 2>/dev/null || true; rm -rf "${1}"; }

if [ -e "${NEW_ENV}" ]; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo "==> FORCE=1: removing existing ${NEW_ENV}"; nuke_env "${NEW_ENV}"
    elif env_ok "${NEW_ENV}"; then
        echo "==> ${NEW_ENV} already exists and is valid — skipping clone (FORCE=1 to recreate)."
    else
        echo "==> ${NEW_ENV} exists but is broken (no bin/python) — removing and recreating."
        nuke_env "${NEW_ENV}"
    fi
fi

if [ ! -e "${NEW_ENV}" ]; then
    # `conda create --clone` is fundamentally unworkable here: the donor was built
    # with mamba, and conda tries to RE-DOWNLOAD packages (its records don't match
    # the conda caches — especially the epoch package x264 '1!161.3030' with '%21' in
    # the url): --offline -> OfflineError, online -> breaks on large CUDA libs
    # (IncompleteRead) and leaves a broken partial. The donor env is already fully
    # local and working -> we just COPY it and fix the prefix. On the donor it was
    # verified: 0 binaries and 0 abs symlinks contain the old prefix -> `cp -a` +
    # `sed` over text is enough (~567 files: shebangs in bin/, *.pc / *Config.sh in
    # lib/, conda-meta/history). Fully offline, no network and no conda caches; `cp -a`
    # preserves permissions, symlinks and hardlinks.
    echo "==> copy-clone: cp -a ${SRC_ENV}  ->  ${NEW_ENV}  (~12 GB, local, no network)"
    if ! cp -a "${SRC_ENV}" "${NEW_ENV}"; then
        echo "ERROR: cp -a failed — removing partial copy ${NEW_ENV}"; rm -rf "${NEW_ENV}"; exit 1
    fi
    echo "==> fixing prefix in text files: ${SRC_ENV} -> ${NEW_ENV}"
    pfx_list="$(mktemp)"
    grep -rlIZ -- "${SRC_ENV}" "${NEW_ENV}" 2>/dev/null > "${pfx_list}" || true
    nfix="$(tr -cd '\0' < "${pfx_list}" | wc -c)"
    xargs -0 -r sed -i "s|${SRC_ENV}|${NEW_ENV}|g" < "${pfx_list}"
    rm -f "${pfx_list}"
    echo "    files fixed: ${nfix}"
    # quick check: no references to the old prefix remain in bin/ (shebangs)
    if grep -rlI -- "${SRC_ENV}" "${NEW_ENV}/bin" >/dev/null 2>&1; then
        echo "WARNING: references to the old prefix remain in ${NEW_ENV}/bin — check manually"
    fi
fi

PY="${NEW_ENV}/bin/python"
PIP="${NEW_ENV}/bin/pip"
env_ok "${NEW_ENV}" || { echo "ERROR: clone did not produce python in ${NEW_ENV}"; exit 1; }

# ---- 2. install cuEquivariance --------------------------------------------
# IMPORTANT: we need NOT only cuequivariance-torch, but also the fused kernels
# cuequivariance-ops-torch-cu12. Without the ops package cuet falls back to the naive
# backend (SegmentedPolynomialNaive), which has no .buffer_num_segments -> mace 0.3.15
# with enable_cueq=True on CUDA (conv_fusion=True is hardcoded there, cannot disable) crashes:
#   AttributeError: 'SegmentedPolynomialNaive' object has no attribute 'buffer_num_segments'
# With the ops package the fused path works on Blackwell sm_120 (cu12 + nvidia-cublas-cu12 12.8),
# lossless. The versions of cuequivariance-torch and -ops-torch-cu12 MUST match.
echo
echo "==> installing ${CUEQ_SPEC} + fused kernels cuequivariance-ops-torch-cu12 (NVIDIA index)"
"${PIP}" install --no-input --upgrade pip >/dev/null
"${PIP}" install --no-input --extra-index-url "${NV_INDEX}" ${CUEQ_SPEC}
CUEQ_VER="$("${PY}" -c 'import cuequivariance_torch as c; print(c.__version__)' 2>/dev/null || true)"
if [ -n "${CUEQ_VER}" ]; then
    echo "    cuequivariance-torch ${CUEQ_VER} -> installing cuequivariance-ops-torch-cu12==${CUEQ_VER}"
    "${PIP}" install --no-input --extra-index-url "${NV_INDEX}" "cuequivariance-ops-torch-cu12==${CUEQ_VER}"
else
    echo "    WARN: could not determine cuequivariance-torch version — installing latest ops"
    "${PIP}" install --no-input --extra-index-url "${NV_INDEX}" cuequivariance-ops-torch-cu12
fi

echo
echo "==> installed cueq packages:"
"${PIP}" list 2>/dev/null | grep -iE "cuequivariance|^mace-torch|^torch|^e3nn|^ase" || true

# ---- 3. verify on GPU (via srun) ------------------------------------------
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
    echo; echo "==> SKIP_VERIFY=1 — skipping GPU verification."
    echo "Done. env: ${NEW_ENV}"
    exit 0
fi

VERIFY_PY="$(mktemp --suffix=_verify_cueq.py)"
cat > "${VERIFY_PY}" <<'PYEOF'
import os, sys, time, traceback, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np

MODEL = os.environ["VERIFY_MODEL"]
HEAD  = os.environ.get("VERIFY_HEAD", "omat_pbe")

import torch
print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)} | "
      f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    print(f"cuequivariance       {getattr(cue, '__version__', '?')}")
    print(f"cuequivariance_torch {getattr(cuet, '__version__', '?')}")
except Exception:
    print("!! cannot import cuequivariance / cuequivariance_torch:")
    traceback.print_exc(); sys.exit(3)

from ase.build import bulk
from mace.calculators import MACECalculator

# deterministic test cells (metal + ionic)
def mk(name, kw, rep, seed):
    a = bulk(name, **kw).repeat(rep)
    a.positions += np.random.default_rng(seed).normal(scale=0.05, size=a.positions.shape)
    return name, a
tests = [
    mk("Cu",   dict(crystalstructure="fcc",      a=3.61), (3,3,3), 0),
    mk("MgO",  dict(crystalstructure="rocksalt", a=4.21), (2,2,2), 5),
]

def calc(enable):
    return MACECalculator(model_paths=MODEL, device="cuda",
                          default_dtype="float64", head=HEAD, enable_cueq=enable)

print("\n== building calculator enable_cueq=False ==")
c_off = calc(False)
print("== building calculator enable_cueq=True  ==")
try:
    c_on = calc(True)
except Exception:
    print("\n!! MACE could not initialize enable_cueq=True:")
    traceback.print_exc()
    print("\nDIAGNOSIS: cueq imports, but MACE 0.3.15 did not agree with this API version "
          "(likely API drift). Pin the version: CUEQ_SPEC='cuequivariance-torch==<compatible>'.")
    sys.exit(3)

def single(c, atoms):
    atoms = atoms.copy(); atoms.calc = c
    return float(atoms.get_potential_energy()), atoms.get_forces()

def timed(c, atoms, n=20, w=3):
    atoms = atoms.copy(); atoms.calc = c
    for _ in range(w): c.calculate(atoms)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): c.calculate(atoms)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

print(f"\n{'structure':>10} {'natoms':>7} {'dE/atom[eV]':>13} {'dFmax[eV/A]':>13} "
      f"{'t_off[ms]':>10} {'t_on[ms]':>10} {'speedup':>8}")
ok = True
for name, atoms in tests:
    e0, f0 = single(c_off, atoms)
    e1, f1 = single(c_on,  atoms)
    de = abs(e0 - e1) / len(atoms)
    df = float(np.abs(f0 - f1).max())
    t0 = timed(c_off, atoms); t1 = timed(c_on, atoms)
    spd = t0 / t1 if t1 else float("nan")
    finite = np.isfinite(e1) and np.isfinite(f1).all()
    lossless = de < 1e-4 and df < 1e-3
    ok = ok and finite and lossless
    print(f"{name:>10} {len(atoms):>7} {de:13.2e} {df:13.2e} "
          f"{t0*1e3:10.1f} {t1*1e3:10.1f} {spd:7.2f}x")

print("\n" + "-"*72)
print("VERDICT:", "PASS — cueq works on Blackwell and is lossless (E/F match)"
      if ok else "FAIL — see deltas above (non-finite or drift > threshold)")
sys.exit(0 if ok else 4)
PYEOF

export VERIFY_MODEL="${MODEL}"
export VERIFY_HEAD="${HEAD}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo
echo "==> GPU verification (cueq OFF vs ON) ..."
rc=0
if command -v srun >/dev/null 2>&1; then
    echo "    via srun --gres=gpu:1 --partition=${ZBL_SLURM_PARTITION:-main}"
    set +e
    srun --partition="${ZBL_SLURM_PARTITION:-main}" --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=00:20:00 \
        "${PY}" "${VERIFY_PY}"
    rc=$?
    set -e
else
    echo "    srun unavailable — running directly (this node has a GPU)"
    set +e; "${PY}" "${VERIFY_PY}"; rc=$?; set -e
fi
rm -f "${VERIFY_PY}"

echo
echo "============================================================================"
case "${rc}" in
  0) echo "OK. env with cuEquivariance: ${NEW_ENV}"
     echo
     echo "Run the cueq bench and compare with the already-captured fp64 baseline:"
     echo "  ${PY} bench/bench_mace.py --model ${MODEL} --device cuda \\"
     echo "      --dtype float64 --head ${HEAD} --enable-cueq \\"
     echo "      --structures bench/runs/structures.extxyz --out bench/runs/cueq.json --label cueq"
     echo "  ${PY} bench/compare_runs.py bench/runs/fp64.json bench/runs/cueq.json"
     echo "  # expect PASS (lossless) + large speedup on symmetric contraction" ;;
  3) echo "WARNING: env created (${NEW_ENV}), BUT cueq did not agree with MACE (see traceback above)."
     echo "    Try a pin:  FORCE=1 CUEQ_SPEC='cuequivariance-torch==<version>' bash $0" ;;
  4) echo "WARNING: cueq started, but the losslessness verification did not pass — inspect the deltas above." ;;
  *) echo "WARNING: verification finished with code ${rc} — see output above." ;;
esac
echo "============================================================================"
exit "${rc}"
