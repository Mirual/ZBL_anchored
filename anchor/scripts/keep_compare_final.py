#!/usr/bin/env python3
"""Canonical anchor-vs-FT comparison on the data where the novelty gate actually fires:
the OOD short-contact user `keep` set. (mixed_test/MPtrj are in-distribution → gate silent →
anchor ≡ vanilla there; that table only proves base-preservation, not the force benefit.)

Uses the FIXED make_calculator (core_zbl=False — foundation owns the deep core; no double-ZBL).
Reports R²/MAE on all keep frames, plus an FT number excluding the 2 sub-0.30 Å frames where
FT's retrained head diverges, for a fairer FT comparison.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

ROOT = Path(os.environ.get("ZBL_IAML_WS", ""))  # TODO: set for your machine (external workspace)
WS = ROOT / "idea_uncertainty_gated_physics_anchor"
if os.environ.get("ZBL_IAML_WS"):
    sys.path.insert(0, str(WS / "raddmg" / "scripts"))
from common_calc import make_calculator  # noqa: E402

KEEP = ROOT / "vasp_tier1/collected/preflight/splits/keep_test.xyz"
FT = ROOT / "mace_mh_mix_clean/results/mace_mh0_mix_clean_F10E1/mace_mh0_mix_clean_F10E1.model"
SHORT = 0.30  # dimer-cache floor; FT diverges below this


def forces(calc, at):
    a = at.copy(); a.calc = calc
    return np.asarray(a.get_forces(), float)


def fmin(at):
    d = neighbor_list("d", at, 5.0)
    return float(d.min()) if len(d) else 9.0


def r2(P, R):
    P = np.concatenate([x.reshape(-1) for x in P]); R = np.concatenate([x.reshape(-1) for x in R])
    return float(1 - ((P - R) ** 2).sum() / ((R - R.mean()) ** 2).sum())


def mae(P, R):
    P = np.concatenate([x.reshape(-1) for x in P]); R = np.concatenate([x.reshape(-1) for x in R])
    return float(np.mean(np.abs(P - R)))


def main():
    van = make_calculator("vanilla")
    anc = make_calculator("vanilla_pairphys")           # FIXED: core_zbl=False
    ft = MACECalculator(model_paths=[str(FT)], device="cuda", default_dtype="float32", head="Default")

    keep = read(str(KEEP), index=":")
    ref, Pv, Pa, Pf, md = [], [], [], [], []
    for at in keep:
        fr = at.arrays.get("REF_forces")
        if fr is None:
            continue
        ref.append(np.asarray(fr, float)); md.append(fmin(at))
        Pv.append(forces(van, at)); Pa.append(forces(anc, at)); Pf.append(forces(ft, at))
    md = np.array(md)
    keepmask = md >= SHORT  # frames where FT does not explode

    out = {
        "set": "keep_test (OOD short-contact user data; novelty gate active 6.9% atoms)",
        "n_frames": len(ref), "n_frames_ge_0.30A": int(keepmask.sum()),
        "all": {
            "vanilla": {"F_R2": r2(Pv, ref), "F_MAE": mae(Pv, ref)},
            "anchor":  {"F_R2": r2(Pa, ref), "F_MAE": mae(Pa, ref)},
            "ft":      {"F_R2": r2(Pf, ref), "F_MAE": mae(Pf, ref)},
        },
        "ge_0.30A": {  # exclude 2 sub-floor frames where FT diverges (fairer to FT)
            "vanilla": {"F_R2": r2([p for p, k in zip(Pv, keepmask) if k], [r for r, k in zip(ref, keepmask) if k]),
                        "F_MAE": mae([p for p, k in zip(Pv, keepmask) if k], [r for r, k in zip(ref, keepmask) if k])},
            "anchor":  {"F_R2": r2([p for p, k in zip(Pa, keepmask) if k], [r for r, k in zip(ref, keepmask) if k]),
                        "F_MAE": mae([p for p, k in zip(Pa, keepmask) if k], [r for r, k in zip(ref, keepmask) if k])},
            "ft":      {"F_R2": r2([p for p, k in zip(Pf, keepmask) if k], [r for r, k in zip(ref, keepmask) if k]),
                        "F_MAE": mae([p for p, k in zip(Pf, keepmask) if k], [r for r, k in zip(ref, keepmask) if k])},
        },
    }
    (WS / "results").mkdir(exist_ok=True)
    json.dump(out, open(WS / "results/anchor_vs_ft_keep.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

    # figure: R² and MAE bars, all + ge0.30 for FT
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    models = [("vanilla", "tab:gray"), ("anchor", "tab:red"), ("ft", "tab:blue")]
    x = np.arange(len(models)); w = 0.6
    r2v = [out["all"][m]["F_R2"] for m, _ in models]
    maev = [out["all"][m]["F_MAE"] for m, _ in models]
    r2v_plot = [max(v, -0.5) for v in r2v]  # clip FT's huge-negative for readability
    ax[0].bar(x, r2v_plot, w, color=[c for _, c in models])
    for xi, v in zip(x, r2v):
        ax[0].text(xi, max(v, -0.5) + 0.02, ("diverges" if v < -1 else f"{v:.3f}"), ha="center", fontsize=9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["vanilla", "anchor", "fine-tune"])
    ax[0].set_title("Force R² vs DFT on keep (↑)  [FT clipped]"); ax[0].axhline(0, color="k", lw=0.5)
    ax[1].bar(x, np.log10(maev), w, color=[c for _, c in models])
    for xi, v in zip(x, maev):
        ax[1].text(xi, np.log10(v) + 0.05, f"{v:.1f}" if v < 1e3 else f"{v:.0e}", ha="center", fontsize=9)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["vanilla", "anchor", "fine-tune"])
    ax[1].set_title("log₁₀ Force MAE vs DFT on keep [eV/Å] (↓)")
    fig.suptitle(f"anchor vs fine-tune on OOD short-contact keep (n={len(ref)}, gate active) — forces")
    fig.tight_layout(); fig.savefig(WS / "figures/anchor_vs_ft_keep.png", dpi=140)
    import shutil
    shutil.copy(WS / "figures/anchor_vs_ft_keep.png", WS / "cool/figures/anchor_vs_ft_keep.png")
    print("→ figures/anchor_vs_ft_keep.png + cool/figures/")


if __name__ == "__main__":
    main()
