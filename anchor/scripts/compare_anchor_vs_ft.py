#!/usr/bin/env python3
"""Head-to-head on the FINE-TUNE's own held-out split: anchor vs fine-tuning vs vanilla.

Evaluates force accuracy (F R², F-MAE vs DFT REF_forces) of three models on
mixed_dataset_clean/mixed_test.xyz (the split the MACE-MH fine-tune was tested on, and did
NOT train on — fair to both FT and anchor):
  - vanilla   : MACE-MH-0 (head mp_pbe_refit_add)
  - anchor    : vanilla + RND-gated pairphys (our method, training-free)
  - ft        : mace_mh0_mix_clean_F10E1 (fine-tuned MACE-MH, head Default)
Split by frame min interatomic distance: distorted (<1.5 Å, where the anchor acts /
user-distorted) vs normal (≥1.5 Å, MPtrj-like base) — shows accuracy-vs-base-preservation.

    nohup python compare_anchor_vs_ft.py &
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(os.environ.get("ZBL_IAML_WS", ""))  # TODO: set for your machine (external workspace)
if os.environ.get("ZBL_IAML_WS"):
    sys.path.insert(0, str(ROOT / "idea_uncertainty_gated_physics_anchor" / "raddmg" / "scripts"))
from common_calc import make_calculator  # noqa: E402

FT = ROOT / "mace_mh_mix_clean/results/mace_mh0_mix_clean_F10E1/mace_mh0_mix_clean_F10E1.model"
TEST = ROOT / "mixed_dataset_clean/data/mixed_test.xyz"
OUTJ = ROOT / "idea_uncertainty_gated_physics_anchor/results/anchor_vs_ft.json"
FIG = ROOT / "idea_uncertainty_gated_physics_anchor/figures/anchor_vs_ft.png"
DIST_CUT = 1.5
MODELS = ("vanilla", "anchor", "ft")
GROUPS = ("all", "distorted", "normal", "mptrj")
# Display-only labels (poster terminology). Keys above stay as stable lookup keys.
GROUP_DISPLAY = {
    "all": "all",
    "distorted": "distorted (compressed OOD)",
    "normal": "in-distribution (baseline)",
    "mptrj": "MPtrj (baseline)",
}
MPTRJ_CANDIDATES = [ROOT / "idea_uncertainty_gated_physics_anchor/results/mptrj1k.xyz",
                    ROOT / "pct_compare_10k/data/mptrj_stratified_10k.xyz"]
MPTRJ_N = 1000


def frame_min_dist(at) -> float:
    d = neighbor_list("d", at, 5.0)
    return float(d.min()) if len(d) else 9.0


def r2(pred, ref) -> float:
    p = np.concatenate([x.reshape(-1) for x in pred]); r = np.concatenate([x.reshape(-1) for x in ref])
    return float(1.0 - ((p - r) ** 2).sum() / ((r - r.mean()) ** 2).sum())


def mae(pred, ref) -> float:
    p = np.concatenate([x.reshape(-1) for x in pred]); r = np.concatenate([x.reshape(-1) for x in ref])
    return float(np.mean(np.abs(p - r)))


def main() -> None:
    calcs = {
        "vanilla": make_calculator("vanilla"),
        "anchor": make_calculator("vanilla_pairphys"),
        "ft": MACECalculator(model_paths=[str(FT)], device="cuda", default_dtype="float32", head="Default"),
    }
    def get_ref(at):
        for kk in ("REF_forces", "forces"):
            if kk in at.arrays:
                return np.asarray(at.arrays[kk], float)
        return None

    def predict(at):
        out = {}
        for m, calc in calcs.items():
            a = at.copy(); a.calc = calc
            try:
                out[m] = np.asarray(a.get_forces(), float)
            except Exception:
                return None
        return out

    pools = {g: {"ref": [], **{m: [] for m in MODELS}} for g in GROUPS}
    nframes = {g: 0 for g in GROUPS}

    def add(at, groups):
        Fref = get_ref(at)
        if Fref is None:
            return
        preds = predict(at)
        if preds is None:
            return
        for g in groups:
            pools[g]["ref"].append(Fref)
            for m in MODELS:
                pools[g][m].append(preds[m])
            nframes[g] += 1

    # (1) FT's held-out mixed_test → all + distorted/normal by min interatomic distance
    frames = read(str(TEST), index=":")
    for k, at in enumerate(frames):
        grp = "distorted" if frame_min_dist(at) < DIST_CUT else "normal"
        add(at, ("all", grp))
        if (k + 1) % 300 == 0:
            print(f"  mixed_test {k+1}/{len(frames)}", flush=True)

    # (2) pure MPtrj base group (foundation home — base-preservation check)
    mptrj_path = next((p for p in MPTRJ_CANDIDATES if p.exists()), None)
    if mptrj_path is not None:
        mp = read(str(mptrj_path), index=":")
        if len(mp) > MPTRJ_N:
            mp = mp[:: max(1, len(mp) // MPTRJ_N)][:MPTRJ_N]
        print(f"  MPtrj from {mptrj_path.name}: {len(mp)} frames")
        for k, at in enumerate(mp):
            add(at, ("mptrj",))
            if (k + 1) % 300 == 0:
                print(f"  mptrj {k+1}/{len(mp)}", flush=True)
    else:
        print("  WARNING: no MPtrj file found → mptrj group empty")

    summary = {"n_frames": nframes, "dist_cut": DIST_CUT}
    print(f"\n{'group':>10} {'n_fr':>5} | " + " ".join(f"{m+' R2':>11}" for m in MODELS)
          + " | " + " ".join(f"{m+' MAE':>10}" for m in MODELS))
    for g in GROUPS:
        if not pools[g]["ref"]:
            continue
        row = {m: dict(F_R2=r2(pools[g][m], pools[g]["ref"]), F_MAE=mae(pools[g][m], pools[g]["ref"]))
               for m in MODELS}
        summary[g] = row
        print(f"{g:>10} {nframes[g]:>5} | " + " ".join(f"{row[m]['F_R2']:>11.3f}" for m in MODELS)
              + " | " + " ".join(f"{row[m]['F_MAE']:>10.3f}" for m in MODELS))
    OUTJ.write_text(json.dumps(summary, indent=2))
    print("\n→", OUTJ)

    # figure: F R² and F-MAE, 3 models × 3 groups
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(GROUPS)); w = 0.26
    colors = {"vanilla": "tab:gray", "anchor": "tab:red", "ft": "tab:blue"}
    for i, m in enumerate(MODELS):
        ax[0].bar(x + (i - 1) * w, [summary.get(g, {}).get(m, {}).get("F_R2", 0) for g in GROUPS],
                  w, label=m, color=colors[m])
        ax[1].bar(x + (i - 1) * w, [summary.get(g, {}).get(m, {}).get("F_MAE", 0) for g in GROUPS],
                  w, label=m, color=colors[m])
    for a, t in zip(ax, ("Force R² vs DFT (↑)", "Force MAE vs DFT [eV/Å] (↓)")):
        a.set_xticks(x); a.set_xticklabels([f"{GROUP_DISPLAY[g]}\n(n={nframes[g]})" for g in GROUPS])
        a.set_title(t); a.legend()
    ax[0].axhline(0, color="k", lw=0.5)
    fig.suptitle("anchor vs fine-tuning vs vanilla — on the FT's held-out mixed_test (forces)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print("→", FIG)


if __name__ == "__main__":
    main()
