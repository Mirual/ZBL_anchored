#!/usr/bin/env bash
set -uo pipefail; export PATH="$(dirname "${ZBL_PYTHON:-python}")":$PATH
cd "$(dirname "$0")/.."
PY=${ZBL_PYTHON:-python}
RUN(){ $PY scripts/md_run.py --system systems/$1.xyz --mode $2 $4 --T $5 --ensemble $3 \
       --steps 2000 --dt 1.0 --log-every 10 --out results/nozbl_${1}_${3}${5}_$6.json 2>&1 | grep -E "COLLAPSED|survived" | tail -1; }
for spec in "compressed_0:nve:3000" "compressed_4:nvt:4000" "hot_1:nvt:3000"; do
  IFS=: read s ens T <<< "$spec"
  echo "### $s $ens T=$T $(date +%H:%M:%S)"
  RUN $s vanilla   $ens ""             $T vanZBL
  RUN $s vanilla   $ens "--disable-zbl" $T vanNOZBL
  RUN $s bornmayer $ens "--disable-zbl" $T bornNOZBL
done
echo "### NOZBL DONE $(date +%H:%M:%S)"
