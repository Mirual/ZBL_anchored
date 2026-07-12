#!/usr/bin/env bash
# Defect formation energies vs DFT/literature: UO2 (fixed 3x3x3, no recombination) + SrTiO3.
# mu-free Frenkel/antisite compared directly to published values. Run via nohup (no own DFT).
set -uo pipefail
export PATH="/path/to/conda-env/bin:$PATH"  # TODO: set for your machine
PY="${ZBL_PYTHON:-python}"
WS="${ZBL_ANCHOR_WS:-/path/to/idea_uncertainty_gated_physics_anchor}"  # TODO: set for your machine
S="$WS/raddmg/scripts"
cd "$WS"

echo "host $(hostname)  pid $$  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)  started $(date -Is)"

echo "=== build defect cells (3x3x3) ==="
$PY "$S/build_defects.py" --host UO2    --nrep 3 --species U O    || { echo FATAL; exit 1; }
$PY "$S/build_defects.py" --host SrTiO3 --nrep 3 --species Sr Ti O || { echo FATAL; exit 1; }

for C in vanilla vanilla_pairphys; do
  echo "=== [$C] T0.2 defects UO2 (3x3x3) ==="
  $PY "$S/t0_defect_static.py" --host UO2    --nrep 3 --calc "$C" || echo "WARN defect UO2 $C"
  echo "=== [$C] T0.2 defects SrTiO3 (3x3x3) ==="
  $PY "$S/t0_defect_static.py" --host SrTiO3 --nrep 3 --calc "$C" || echo "WARN defect SrTiO3 $C"
done

echo "=== compare defects to literature ==="
$PY "$S/compare_defects.py" || true
echo "=== DONE $(date -Is) ==="
