#!/usr/bin/env python3
"""STUB: fine-tune MACE-MH-0 or MACE-MH-1 on the absolute-energy 26_02_5and8 splits.

Currently disabled — review zero-shot metrics first
(see results/metrics.json from scripts/03_evaluate.py).
Promote by removing the SystemExit gate at the top of main().

Recipe is the working `vasp_done` config (lr=5e-3, batch=4) — same as
gfnff_delta/scripts/03_train_mace.py — applied to the absolute energy
target instead of the GFN-FF delta. innp/finetune/02_train_mace.py's
lr=1e-4 + energy_weight=1000 collapses on this dataset; do NOT mirror it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
PRETRAINED = WS / "pretrained"
RESULTS = WS / "results"
LOGS = WS / "logs"

# Absolute-energy splits from the cleaned innp dataset.
TRAIN = WS.parent / "innp" / "finetune" / "stage1_clean" / "cleaned_train.xyz"
VAL = WS.parent / "innp" / "finetune" / "stage1_clean" / "cleaned_val.xyz"
TEST = WS.parent / "innp" / "finetune" / "stage1_clean" / "cleaned_test.xyz"

LR = 5e-3
BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
MAX_EPOCHS = 200
PATIENCE = 50
SCHEDULER_PATIENCE = 20
LR_FACTOR = 0.5
ENERGY_WEIGHT = 10.0
FORCES_WEIGHT = 1.0
STRESS_WEIGHT = 0.0
EMA_DECAY = 0.99
SEED = 42
DTYPE = "float32"
DEVICE = "cuda"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mh0", "mh1"], default="mh0")
    args = p.parse_args()

    foundation = PRETRAINED / f"mace-{args.model.replace('mh', 'mh-')}.model"
    if not foundation.exists():
        sys.exit(f"ERROR: {foundation} missing — run 01_download_models.py first.")

    run_name = f"mace_{args.model}_zbl_finetune"
    out_dir = RESULTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mace_run_train",
        f"--name={run_name}",
        f"--foundation_model={foundation}",
        "--foundation_head=mp_pbe_refit_add",
        "--multiheads_finetuning=False",
        f"--train_file={TRAIN}",
        f"--valid_file={VAL}",
        f"--test_file={TEST}",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        "--E0s=average",
        f"--device={DEVICE}",
        f"--default_dtype={DTYPE}",
        f"--batch_size={BATCH_SIZE}",
        f"--valid_batch_size={VALID_BATCH_SIZE}",
        f"--lr={LR}",
        f"--max_num_epochs={MAX_EPOCHS}",
        f"--patience={PATIENCE}",
        f"--lr_factor={LR_FACTOR}",
        f"--scheduler_patience={SCHEDULER_PATIENCE}",
        f"--energy_weight={ENERGY_WEIGHT}",
        f"--forces_weight={FORCES_WEIGHT}",
        f"--stress_weight={STRESS_WEIGHT}",
        "--ema",
        f"--ema_decay={EMA_DECAY}",
        "--amsgrad",
        "--scaling=rms_forces_scaling",
        "--save_cpu",
        f"--seed={SEED}",
        f"--results_dir={out_dir}",
    ]
    print(" ".join(cmd))
    sys.exit(subprocess.run(cmd, cwd=str(out_dir)).returncode)


if __name__ == "__main__":
    main()
