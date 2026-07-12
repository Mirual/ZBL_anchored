#!/usr/bin/env bash
set -uo pipefail; export PATH="$(dirname "${ZBL_PYTHON:-python}")":$PATH
cd "$(dirname "$0")/.."
PY=${ZBL_PYTHON:-python}
for sys_T in "compressed_0:1200" "normal_0:600"; do
  s=${sys_T%%:*}; T=${sys_T#*:}
  for m in vanilla bornmayer pairphys; do
    echo "### $s $m T=$T $(date +%H:%M:%S)"
    $PY scripts/md_run.py --system systems/$s.xyz --mode $m --T $T --ensemble nvt \
        --steps 2000 --dt 1.0 --log-every 10 --out results/pilot_${s}_${m}.json 2>&1 | grep -E "COLLAPSED|survived" | tail -1
  done
done
echo "### PILOT DONE $(date +%H:%M:%S)"
