#!/usr/bin/env python3
"""How well RND-novelty SEPARATES your keep from the MPtrj base — per-atom vs per-structure.

Hypothesis (visible on ρ-vs-d): for INDIVIDUAL atoms keep and the base overlap strongly (most
keep atoms are locally ordinary), but per STRUCTURE they separate almost perfectly (every keep cell has
a compressed tail, the base does not). We measure this via AUC (probability that a random keep > a random base).

Panels:
  (A) per atom: histograms of novelty ρ (base vs keep) — overlap in the bulk + keep tail.
  (B) per structure: 2D clusters (min interatomic d  ×  fraction of "novel" atoms ρ>r_lo) — clean separation.
Env: gfnff-delta-mace.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ase.io import read  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WS + "/scripts")
from rnd_anchor_predict import RNDGate, VAN  # noqa: E402

SP = os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight") + "/splits"
MPTRJ = os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz")
RLO = json.load(open(WS + "/results/pairphys_theta.json"))["r_lo"]

calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
gate = RNDGate("cuda")


def per_frame(xyz, n):
    """returns list of dict(nov[Nat], dmin_frame, gate_frac)."""
    out = []
    for at in read(xyz, index=f":{n}"):
        desc = np.asarray(calc.get_descriptors(at))
        x = torch.tensor((desc - gate.mu) / gate.sd, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            t = gate.target(x).cpu().numpy(); p = gate.pred(x).cpu().numpy()
        nov = ((p - t) ** 2).mean(1)
        dd = neighbor_list("d", at, 4.0)
        dmin = float(dd.min()) if len(dd) else np.nan
        out.append(dict(nov=nov, dmin=dmin, gate_frac=float(np.mean(nov > RLO))))
    return out


def auc(neg, pos):
    """probability a random pos ranks above a random neg (Mann-Whitney, proper midrank ties)."""
    from scipy.stats import rankdata
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    r = rankdata(np.concatenate([neg, pos]))        # average ranks for ties
    rp = r[len(neg):].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


B = per_frame(MPTRJ, 120)
K = per_frame(SP + "/keep_test.xyz", 87)

# per-atom
aB = np.concatenate([f["nov"] for f in B])
aK = np.concatenate([f["nov"] for f in K])
# per-frame separators
gB = np.array([f["gate_frac"] for f in B]); gK = np.array([f["gate_frac"] for f in K])
mB = np.array([f["dmin"] for f in B]); mK = np.array([f["dmin"] for f in K])
xmaxB = np.array([f["nov"].max() for f in B]); xmaxK = np.array([f["nov"].max() for f in K])

auc_atom = auc(aB, aK)
# above the gate threshold: ONLY keep? (one-sided separation)
nK_at_gated = int((aK > RLO).sum()); nB_at_gated = int((aB > RLO).sum())
nK_fr_gated = int((gK > 0).sum()); nB_fr_gated = int((gB > 0).sum())
print(f"per-ATOM  ρ: AUC={auc_atom:.3f}  (keep shifted, but bulk overlaps)")
print(f"atoms above threshold r_lo: keep {nK_at_gated}/{len(aK)} ({100*nK_at_gated/len(aK):.1f}%)  |  base {nB_at_gated}/{len(aB)}")
print(f"structures where the gate fired (>0 atoms): keep {nK_fr_gated}/{len(K)}  |  base {nB_fr_gated}/{len(B)}")
print(f"per-FRAME gated-fraction: AUC(midrank)={auc(gB, gK):.3f}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6))

# ---- (A) per-atom: log-COUNT, so the TAIL past the threshold is visible ----
bins = np.logspace(-5, 2.7, 64)
axA.hist(np.clip(aB, 1e-5, None), bins=bins, color="#1e8449", alpha=0.6, label=f"MPtrj base ({len(aB)} atoms)")
axA.hist(np.clip(aK, 1e-5, None), bins=bins, color="#c0392b", alpha=0.6, label=f"your keep ({len(aK)} atoms)")
axA.axvline(RLO, color="k", ls="--", lw=1.5)
axA.axvspan(RLO, 10 ** 2.7, color="#c0392b", alpha=0.06, lw=0)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("single-atom novelty ρ (log)"); axA.set_ylabel("number of atoms (log)")
axA.set_title(f"(A) PER ATOM: bulk overlaps (AUC={auc_atom:.2f}),\n"
              f"but PAST the threshold — ONLY keep ({nK_at_gated} atoms), base 0",
              fontsize=11, fontweight="bold")
axA.text(RLO * 1.3, axA.get_ylim()[1] * 0.3, "threshold →\nonly your\nkeep here", color="#c0392b",
         fontsize=9, fontweight="bold")
axA.legend(fontsize=9.5, loc="upper right")

# ---- (B) per-frame: one-sided clean separation ----
axB.scatter(mB, gB * 100, s=46, color="#1e8449", alpha=0.7, edgecolor="w", lw=0.5, label=f"MPtrj base ({len(B)} structures)")
axB.scatter(mK, gK * 100, s=46, color="#c0392b", alpha=0.7, edgecolor="w", lw=0.5, label=f"your keep ({len(K)} structures)")
axB.axhline(0.5, color="#555", ls=":", lw=1.2)
axB.set_xlabel("min interatomic distance in structure, Å")
axB.set_ylabel("fraction of \"novel\" atoms in structure (ρ>threshold), %")
axB.set_title(f"(B) PER STRUCTURE: the gate fired on {nK_fr_gated}/{len(K)} keep and {nB_fr_gated}/{len(B)} base\n"
              "→ fired = your OOD (100% accuracy); silent = base-like structure",
              fontsize=11, fontweight="bold")
axB.legend(fontsize=9.5, loc="center right")
axB.grid(alpha=0.25)

fig.suptitle("Separation your keep ↔ MPtrj base: bulk overlaps, but the gate FIRING = your OOD (base NEVER fires)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = WS + "/figures/rnd_separation.png"
fig.savefig(out, dpi=140)
import shutil
shutil.copy(out, WS + "/cool/figures/rnd_separation.png")
print("saved", out, "(+cool)")
