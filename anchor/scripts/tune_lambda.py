#!/usr/bin/env python3
"""Layer 3 sanity: one pass caches (F_van, F_ref, F_corr, E_van, E_ref, E_corr) on
keep/u200/MPtrj-500, then an analytical sweep of the correction scale λ (F=F_van+λ·F_corr).
Question: is there a λ where keep is better AND u200/MPtrj ≈ vanilla?  If not — Layer 3 is not enough, a gate is needed.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_physics import DimerCache, pair_correction

VAN = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))


def cache_set(frames, calc, dc):
    out = []
    for at in frames:
        fr = at.arrays.get("REF_forces")
        if fr is None:
            continue
        er = float(at.info.get("REF_energy", at.info.get("energy", 0.0)))
        at.calc = calc
        fv = np.asarray(at.get_forces()); ev = float(at.get_potential_energy())
        ec, fc = pair_correction(at, dc)
        out.append((fv, np.asarray(fr), fc, ev, er, ec, len(at)))
    return out


def metrics(cached, lam):
    Fr, Fp, dEpa, Er, Ep = [], [], [], [], []
    for fv, fr, fc, ev, er, ec, nat in cached:
        Fr.append(fr.reshape(-1)); Fp.append((fv + lam * fc).reshape(-1))
        dEpa.append(abs((ev + lam * ec) - er) / nat); Er.append(er); Ep.append(ev + lam * ec)
    Fr = np.concatenate(Fr); Fp = np.concatenate(Fp)
    Er = np.array(Er); Ep = np.array(Ep)
    fR2 = 1 - ((Fr - Fp) ** 2).sum() / ((Fr - Fr.mean()) ** 2).sum()
    eR2 = 1 - ((Er - Ep) ** 2).sum() / ((Er - Er.mean()) ** 2).sum()
    return np.abs(Fp - Fr).mean(), fR2, float(np.mean(dEpa)) * 1000, eR2


def main():
    calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
    dc = DimerCache(calc)
    sets = {
        "u200": cache_set(read(PRE/"splits"/"u200_test.xyz", index=":"), calc, dc),
        "keep": cache_set(read(PRE/"splits"/"keep_test.xyz", index=":"), calc, dc),
        "mptrj": cache_set(read(os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), index="0:120"), calc, dc),
    }
    print(f"pairs cached: {len(dc._cache)}\n")
    print(f"{'λ':>7s} | " + " | ".join(f"{s:^32s}" for s in sets))
    print(f"{'':7s} | " + " | ".join(f"{'F MAE':>7s} {'F R²':>7s} {'E MAE':>7s} {'E R²':>7s}" for _ in sets))
    print("-" * 120)
    for lam in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        row = f"{lam:7.3f} | "
        for s in sets:
            fM, fR, eM, eR = metrics(sets[s], lam)
            row += f"{fM:7.2f} {fR:7.3f} {eM:7.0f} {eR:7.2f} | "
        print(row)


if __name__ == "__main__":
    main()
