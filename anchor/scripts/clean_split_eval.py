#!/usr/bin/env python3
"""Group-aware (composition-disjoint) keep evaluation — removes the near-duplicate leakage
of the random-frame keep_train/keep_test split (all 23 compositions were shared).

Protocol:
  1. pool ALL keep frames (train+val+test);
  2. split by chemical formula → CALIB vs TEST with DISJOINT formulas (no shared composition);
  3. calibrate (r_lo, λ) on CALIB (min F-MAE s.t. base u200 preserved) — same objective as tune_pairphys;
  4. report vanilla vs pairphys F-MAE and F R² on the TEST group (truly-held-out chemistry).

RND novelty is MPtrj-only (clean) and per-pair physics is data-free, so this isolates the only
leakage channel (the 2 calibrated scalars) and gives an unbiased keep number.

    python clean_split_eval.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune_pairphys as T                       # noqa: E402  (cache, corr_F, fmae, maxdf)
from pair_physics import DimerCache             # noqa: E402
from rnd_anchor_predict import RNDGate          # noqa: E402

PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight")) / "splits"
VAN = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
OUT = HERE.parents[0] / "results" / "clean_split_eval.json"


def r2(records, r_lo, r_hi, lam):
    pred = np.concatenate([(r["fv"] + T.corr_F(r, r_lo, r_hi, lam)).reshape(-1) for r in records])
    ref = np.concatenate([r["fr"].reshape(-1) for r in records])
    ss_res = float(((pred - ref) ** 2).sum())
    ss_tot = float(((ref - ref.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def disjoint_split(frames):
    groups = defaultdict(list)
    for k, at in enumerate(frames):
        groups[at.get_chemical_formula()].append(k)
    calib, test, nc, nt = set(), set(), 0, 0
    for g in sorted(groups, key=lambda g: -len(groups[g])):   # big groups first, to smaller side
        if nc <= nt:
            calib |= set(groups[g]); nc += len(groups[g])
        else:
            test |= set(groups[g]); nt += len(groups[g])
    return sorted(calib), sorted(test), groups


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(model_paths=[VAN], device=dev, default_dtype="float32", head="mp_pbe_refit_add")
    gate = RNDGate(dev); dc = DimerCache(calc)

    frames = []
    for f in ("keep_train.xyz", "keep_val.xyz", "keep_test.xyz"):
        p = PRE / f
        if p.exists():
            frames += read(str(p), index=":")
    ci, ti, groups = disjoint_split(frames)
    calib_forms = sorted({frames[k].get_chemical_formula() for k in ci})
    test_forms = sorted({frames[k].get_chemical_formula() for k in ti})
    assert not (set(calib_forms) & set(test_forms)), "formula leak!"
    print(f"pool={len(frames)} frames, {len(groups)} formulas → "
          f"CALIB {len(ci)} frames/{len(calib_forms)} formulas | TEST {len(ti)} frames/{len(test_forms)} formulas")
    print(f"  test formulas (disjoint): {test_forms}")

    C = T.cache([frames[k] for k in ci], calc, gate, dc)
    Te = T.cache([frames[k] for k in ti], calc, gate, dc)
    base = T.cache(read(str(PRE / "u200_train.xyz"), index=":"), calc, gate, dc)

    base0 = T.fmae(base, 1, 2, 0)
    EPS, MAXB = 0.01, 1.0
    best = None
    print(f"\n{'r_lo':>6} {'λ':>6} | {'calib MAE':>9} {'Δbase':>9} {'maxΔF_b':>8} OK")
    for r_lo in (0.05, 0.1):
        r_hi = 10 * r_lo
        for lam in (0.4, 0.8, 1.6, 3.0, 6.0, 12.0):
            km = T.fmae(C, r_lo, r_hi, lam)
            bm = T.fmae(base, r_lo, r_hi, lam)
            mdf = T.maxdf(base, r_lo, r_hi, lam)
            ok = (bm - base0) <= EPS and mdf <= MAXB
            print(f"{r_lo:6.3f} {lam:6.2f} | {km:9.3f} {bm-base0:+9.4f} {mdf:8.3f} {'✓' if ok else ''}")
            if ok and (best is None or km < best[-1]):
                best = (r_lo, r_hi, lam, km)
    r_lo, r_hi, lam, _ = best
    print(f"\nchosen on CALIB: r_lo={r_lo} λ={lam}")

    # ---- unbiased evaluation on the disjoint TEST group ----
    van_mae = T.fmae(Te, 1, 2, 0)
    pp_mae = T.fmae(Te, r_lo, r_hi, lam)
    van_r2 = r2(Te, 1, 2, 0)          # corr=0 → vanilla
    pp_r2 = r2(Te, r_lo, r_hi, lam)
    res = dict(pool=len(frames), n_formulas=len(groups),
               calib_frames=len(ci), test_frames=len(ti),
               calib_formulas=calib_forms, test_formulas=test_forms,
               theta=dict(r_lo=r_lo, r_hi=r_hi, lam=lam),
               test_vanilla=dict(F_MAE=van_mae, F_R2=van_r2),
               test_pairphys=dict(F_MAE=pp_mae, F_R2=pp_r2))
    OUT.write_text(json.dumps(res, indent=2))

    print("\n=== UNBIASED keep eval on composition-disjoint TEST group ===")
    print(f"  vanilla : F-MAE {van_mae:.3f}  F R² {van_r2:.3f}")
    print(f"  pairphys: F-MAE {pp_mae:.3f}  F R² {pp_r2:.3f}   "
          f"(ΔMAE {(pp_mae/van_mae-1)*100:+.1f}%, ΔR² {pp_r2-van_r2:+.3f})")
    print("  (compare to composition-LEAKED random-frame split: vanilla 0.65 → pairphys 0.83)")
    print("→", OUT)


if __name__ == "__main__":
    main()
