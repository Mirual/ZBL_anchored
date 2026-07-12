#!/usr/bin/env bash
set -uo pipefail; export PATH="${ZBL_CONDA_BIN:-/path/to/conda/env/bin}:$PATH"  # TODO: set for your machine
cd "$(dirname "$0")"
PY="${ZBL_PYTHON:-python}"
CACHE=results/dimer_tables/snapshot_diatomics.pkl
echo "### vanilla $(date +%H:%M:%S)"
$PY scripts/mlip_arena_diatomics.py --mode vanilla --cache $CACHE --out results/diatomics_vanilla.json 2>&1 | grep -E "elements|→" | tail -1
echo "### pairphys $(date +%H:%M:%S)"
$PY scripts/mlip_arena_diatomics.py --mode pairphys --cache $CACHE --out results/diatomics_pairphys.json 2>&1 | grep -E "elements|→" | tail -1
echo "### DONE $(date +%H:%M:%S)"
