#!/usr/bin/env python3
"""Pure-ZBL baseline: vanilla DPA-3.1-3M-FT + built-in ZBL-switching (sw=[0.5,1.5]), WITHOUT fine-tuning.

Checkpoint `dpa-3.1-3m-ft-zbl-init.pth` (DPZBLModel wrapper around foundation weights) — the canonical 'pure ZBL'
on DPA. No RND gate / per-pair residual: just the model's raw E/F with ZBL. Contrast to our anchor —
shows what the GATE gives (anchor preserves the base; bolt-on ZBL breaks it — see memory vanilla_zbl verdict).
Writes results/zbl_<cond>.json (same schema as vanilla_/anchor_). Descriptors not needed → fast.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from ase.io import read
from deepmd.calculator import DP
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpa_common import SPLITS, MPTRJ

RES = Path(__file__).resolve().parents[1] / "results"
# TODO: set for your machine
ZBL_INIT = os.environ.get("ZBL_DPA_ZBL_INIT", "/path/to/dpa-3.1-3m-ft-zbl-init.pth")
COND = {"keep_test": (f"{SPLITS}/keep_test.xyz", ":"),
        "u200_test": (f"{SPLITS}/u200_test.xyz", ":"),
        "mptrj": (MPTRJ, "0:400")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", required=True, choices=list(COND))
    args = p.parse_args()
    calc = DP(model=ZBL_INIT)
    path, sl = COND[args.cond]
    rows, nsk = [], 0
    for k, at in enumerate(read(path, index=sl, format="extxyz")):
        er = float(at.info.get("REF_energy", at.info.get("energy", 0.0)))
        fr = at.arrays.get("REF_forces")
        fr = fr.tolist() if fr is not None else None
        at.calc = calc
        try:
            ep = float(at.get_potential_energy()); fp = np.asarray(at.get_forces()).tolist()
        except Exception:
            nsk += 1; ep, fp = None, None
        rows.append({"idx": k, "n_atoms": len(at), "E_ref": er, "E_pred": ep, "F_ref": fr, "F_pred": fp})
    (RES / f"zbl_{args.cond}.json").write_text(json.dumps({"frames": rows}, indent=2))
    print(f"zbl {args.cond}: {k+1} frames, skipped {nsk} → zbl_{args.cond}.json")


if __name__ == "__main__":
    main()
