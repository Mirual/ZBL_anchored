#!/usr/bin/env bash
set -uo pipefail; export PATH="$(dirname "${ZBL_PYTHON:-python}")":$PATH
cd "$(dirname "$0")/.."
PY=${ZBL_PYTHON:-python}
# NVE energy-conservation on normal (anchor should NOT introduce drift); NVE high-T collapse stress on compressed
declare -a RUNS=(
 "normal_0:nve:300"
 "compressed_0:nve:2500"
 "compressed_4:nvt:3000"
 "hot_1:nvt:2500"
)
for spec in "${RUNS[@]}"; do
  IFS=: read s ens T <<< "$spec"
  for m in vanilla bornmayer pairphys; do
    echo "### $s $m $ens T=$T $(date +%H:%M:%S)"
    $PY scripts/md_run.py --system systems/$s.xyz --mode $m --T $T --ensemble $ens \
        --steps 2000 --dt 1.0 --log-every 10 --out results/stress_${s}_${ens}_${m}.json 2>&1 | grep -E "COLLAPSED|survived" | tail -1
  done
done
echo "### STRESS DONE $(date +%H:%M:%S)"
