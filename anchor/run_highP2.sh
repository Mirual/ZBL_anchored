#!/usr/bin/env bash
set -uo pipefail; export PATH="${ZBL_CONDA_BIN:-/path/to/conda/env/bin}:$PATH"  # TODO: set for your machine
cd "$(dirname "$0")"
PY="${ZBL_PYTHON:-python}"
SYS=md_stability/systems
for s in compressed_0 compressed_3 compressed_4; do
  echo "### $s $(date +%H:%M:%S)"
  $PY scripts/mlip_arena_highP.py --mode vanilla  --system $SYS/$s.xyz --out results/highP2_${s}_vanZBL.json  2>&1 | grep -E "CRASH|survived" | tail -1
  $PY scripts/mlip_arena_highP.py --mode pairphys --system $SYS/$s.xyz --out results/highP2_${s}_pairphysZBL.json 2>&1 | grep -E "CRASH|survived" | tail -1
done
echo "### DONE $(date +%H:%M:%S)"
