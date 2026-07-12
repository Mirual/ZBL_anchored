#!/usr/bin/env bash
#SBATCH --job-name=mlip_md
#SBATCH --partition=${ZBL_SLURM_PARTITION:-main}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=md_%j.out
# ---------------------------------------------------------------------------
# Production-MD template: MLIP (MACE / DPA / anchor) in LAMMPS via fix external.
# Edit CALC/DATA/ELEMS/CMDS for your task.
# ---------------------------------------------------------------------------
set -u
LMP_DIR=${ZBL_LMP_DIR:-/path/to/lammps}  # TODO: set for your machine
export PYTHONPATH=$LMP_DIR/pylibs
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

CALC=mace                 # mace | mace-anchor | dpa | dpa-anchor
DATA=my_system.data       # LAMMPS data file (see scripts/xyz2data.py)
ELEMS="Cu"                # elements in the order of data types: "O Ti"
CMDS=cmds_nvt_dump.in     # your MD scenario (without pair_style!)

${ZBL_PYTHON:-python} $LMP_DIR/scripts/mlip_fixext.py \
    --calc $CALC --data $DATA --elems $ELEMS \
    --mode custom --cmds $CMDS --out result.json

# Fastest variant for CALC=mace (compile+cueq, mace 0.3.16), instead of the line above:
# export PYTHONPATH=$LMP_DIR/pylmp
# export TORCHINDUCTOR_COMPILE_THREADS=1
# LD_LIBRARY_PATH=${ZBL_CONDA_ENV:-/path/to/conda-env}/lib \
# ${ZBL_PYTHON_MACE:-python} $LMP_DIR/scripts/mlip_fixext.py \
#     --calc mace --compile --data $DATA --elems $ELEMS \
#     --mode custom --cmds $CMDS --out result.json
