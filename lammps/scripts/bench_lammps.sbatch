#!/usr/bin/env bash
#SBATCH --job-name=mliplmp
#SBATCH --partition=${ZBL_SLURM_PARTITION:-main}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# Bench MLIP-in-LAMMPS (fix external): ms/step NVT-MD + NVE-drift for anchor.
# Within a single sbatch allocation (survives Claude Code restarts). Resumable:
# skips combinations for which runs/bench_<tag>.json already exists.
# ---------------------------------------------------------------------------
set -u
LMP_DIR=${ZBL_LMP_DIR:-/path/to/lammps}  # TODO: set for your machine
PY=${ZBL_PYTHON:-python}
export PYTHONPATH=$LMP_DIR/pylibs
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd $LMP_DIR/dpa_smoke
mkdir -p $LMP_DIR/runs

run_bench () {  # tag calc data elems... (steps=200 nvt 300K)
  local tag=$1 calc=$2 data=$3; shift 3
  local out=$LMP_DIR/runs/bench_${tag}.json
  if [ -f "$out" ]; then echo "SKIP $tag"; return; fi
  echo "=== BENCH $tag ==="
  $PY $LMP_DIR/scripts/mlip_fixext.py --calc "$calc" --data "$data" \
      --elems "$@" --mode nvt --temp 300 --steps 200 --out "$out" \
      --log $LMP_DIR/runs/log.$tag 2>&1 | tail -2
}

# --- speed: MACE (cueq vs e3nn), DPA; 256 / 864 / 2048 atoms ---
run_bench mace_cueq_256   mace data_cu256.lammps  Cu
run_bench mace_cueq_864   mace data_cu864.lammps  Cu
run_bench mace_cueq_2048  mace data_cu2048.lammps Cu
run_bench dpa_256         dpa  data_cu256.lammps  Cu
run_bench dpa_864         dpa  data_cu864.lammps  Cu
run_bench dpa_2048        dpa  data_cu2048.lammps Cu

# e3nn comparison (one size, to show the cueq gain)
out=$LMP_DIR/runs/bench_mace_e3nn_864.json
if [ ! -f "$out" ]; then
  echo "=== BENCH mace_e3nn_864 ==="
  $PY $LMP_DIR/scripts/mlip_fixext.py --calc mace --data data_cu864.lammps \
      --elems Cu --no-cueq --mode nvt --temp 300 --steps 200 --out "$out" \
      --log $LMP_DIR/runs/log.mace_e3nn_864 2>&1 | tail -2
fi

# --- anchor overhead on normal dynamics (gate silent) ---
run_bench mace_anchor_256 mace-anchor data_cu256.lammps Cu
run_bench dpa_anchor_256  dpa-anchor  data_cu256.lammps Cu

# --- NVE-drift: anchor on a compressed structure (gate active) ---
for pair in "mace_anchor_nve:mace-anchor" "dpa_anchor_nve:dpa-anchor"; do
  tag=${pair%%:*}; calc=${pair##*:}
  out=$LMP_DIR/runs/bench_${tag}.json
  if [ -f "$out" ]; then echo "SKIP $tag"; continue; fi
  echo "=== NVE $tag ==="
  $PY $LMP_DIR/scripts/mlip_fixext.py --calc "$calc" \
      --data $LMP_DIR/anchor/data_compressed0.lammps --elems O Ti \
      --mode nve --temp 600 --steps 500 --timestep 0.0005 --out "$out" \
      --log $LMP_DIR/runs/log.$tag 2>&1 | tail -2
done

echo "ALL_DONE"
