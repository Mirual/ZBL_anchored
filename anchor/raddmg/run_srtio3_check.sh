#!/usr/bin/env bash
# SrTiO3 literature cross-check: B0 + threshold displacement energy E_d (vanilla vs pairphys)
# vs published DFT-MD / experiment. Run via nohup (no DFT of our own).
set -uo pipefail
export PATH="/path/to/conda-env/bin:$PATH"  # TODO: set for your machine
PY="${ZBL_PYTHON:-python}"
WS="${ZBL_ANCHOR_WS:-/path/to/idea_uncertainty_gated_physics_anchor}"  # TODO: set for your machine
S="$WS/raddmg/scripts"
cd "$WS"

echo "host $(hostname)  pid $$  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)  started $(date -Is)"

echo "=== build SrTiO3 host (relax) ==="
$PY "$S/build_hosts.py" --hosts SrTiO3 || { echo "FATAL: build_hosts"; exit 1; }

EGRID="15 20 30 40 50 65 80 100"
for C in vanilla vanilla_pairphys; do
  echo "=== [$C] T0.1 EOS SrTiO3 (B0) ==="
  $PY "$S/t0_eos_elastic.py" --hosts SrTiO3 --calc "$C" || echo "WARN: eos $C"
  echo "=== [$C] E_d recoil sweep SrTiO3 (Sr,Ti,O x <100><110><111> x ${EGRID} eV, 4x4x4) ==="
  $PY "$S/t0_recoil_sweep.py" --host SrTiO3 --calc "$C" --species Sr Ti O \
      --dirs 100 110 111 --energies $EGRID --nrep 4 --t-max-fs 1000 --maxsteps 20000 \
      || echo "WARN: recoil $C"
done

echo "=== compare E_d to literature (DFT-MD / exp) ==="
$PY "$S/compare_srtio3_Ed.py" || true
echo "=== DONE $(date -Is) ==="
