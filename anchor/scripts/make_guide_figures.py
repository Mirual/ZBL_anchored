#!/usr/bin/env python3
"""Explanatory figures for the guide: (A) forward-pass architecture, (B) gate on real data, (C) 3 zones of V(r)."""
import os, sys; from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, "scripts")
FIG = Path("figures")

# ============ FIG A: architecture ============
fig, ax = plt.subplots(figsize=(13.5, 6.5)); ax.axis("off"); ax.set_xlim(0, 16.5); ax.set_ylim(0, 10)
def box(x, y, w, h, t, c, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=c, ec="k", lw=1.3))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, fontweight="bold")
def arr(x1, y1, x2, y2, c="#34495e"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.5, color=c))
box(0.3, 4.2, 2.2, 1.6, "structure\n(atoms,\ncoordinates)", "#ecf0f1")
# upper branch — GNN
box(3.2, 7.4, 3.0, 1.5, "MACE-MH-0\nGNN (foundation)", "#d6eaf8")
box(6.9, 7.4, 2.7, 1.5, "E_GNN,  F_GNN\n(+ built-in ZBL)", "#d6eaf8")
# lower branch — anchor
box(3.2, 5.0, 3.0, 1.3, "per-atom\ndescriptors [256]", "#fdebd0")
box(3.2, 2.9, 3.0, 1.3, "RND novelty\nρ = ‖pred−target‖²", "#fdebd0")
box(6.9, 2.9, 2.7, 1.3, "gate  w = smoothstep(ρ)\nw≈0 normal · w≈1 OOD", "#fde3cf")
box(3.2, 0.6, 3.0, 1.3, "close pairs (i,j)\n→ V_phys(r): ZBL core\n+ bonded-residual", "#d5f5e3", fs=8)
box(6.9, 0.6, 2.7, 1.3, "Σ w·V_phys\n(E_corr, F_corr)", "#d5f5e3")
# merge
box(10.4, 3.9, 3.4, 2.1, "E = E_GNN + Σ w(ρ)·V_phys\nF = F_GNN + Σ w(ρ)·F_phys", "#f9e79f", fs=9.5)
box(14.2, 4.3, 2.0, 1.4, "final\nE, F", "#fadbd8")
arr(2.5, 5.0, 3.2, 8.0); arr(2.5, 5.0, 3.2, 5.6); arr(2.5, 5.0, 3.2, 1.2)
arr(6.2, 8.1, 6.9, 8.1); arr(6.2, 5.6, 4.7, 4.2); arr(4.7, 5.0, 4.7, 4.2)  # desc->rnd
arr(6.2, 3.5, 6.9, 3.5); arr(6.2, 1.2, 6.9, 1.2)
arr(8.3, 3.5, 9.0, 1.6); arr(9.6, 8.1, 10.4, 5.6); arr(9.6, 1.2, 10.4, 4.4)  # gate&corr -> merge
arr(13.8, 5.0, 14.2, 5.0)
ax.annotate("the gate multiplies physics by w → in the normal regime (w≈0) the correction is OFF → base untouched;\n"
            "for compressed/OOD (w≈1) physics is ON. GNN weights are left intact (post-hoc).",
            (0.3, 9.4), fontsize=9, color="#555", fontweight="bold")
fig.suptitle("Architecture: foundation GNN + (RND gate × physical anchor) at the OUTPUT, without retraining",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"guide_architecture.png", dpi=130); plt.close(fig)
print("saved guide_architecture.png")

# ============ FIG B: gate on real data (RND novelty) ============
from mace.calculators import MACECalculator
from rnd_anchor_predict import RNDGate
from anchor_predict import smoothstep
from ase.io import read
VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
g = RNDGate("cuda")
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight")) / "splits"
def nov(path, sl):
    return np.concatenate([g.novelty(calc.get_descriptors(a)) for a in read(path, index=sl)])
nk = nov(PRE/"keep_test.xyz", ":")
nu = nov(PRE/"u200_test.xyz", ":")
nm = nov(os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), "0:200")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
bins = np.logspace(-5, 2.8, 60)
for d, lab, c in [(nm, "MPtrj (baseline)", "#34495e"), (nu, "weakly distorted (target)", "#2980b9"), (nk, "distorted (compressed OOD)", "#c0392b")]:
    ax[0].hist(np.clip(d, 1e-5, None), bins=bins, alpha=.55, label=lab, color=c)
ax[0].axvspan(0.05, 0.5, color="#f9e79f", alpha=.5); ax[0].axvline(0.05, color="k", ls="--", lw=1)
ax[0].annotate("gate threshold\nr_lo=0.05", (0.06, ax[0].get_ylim()[1]*0.5), fontsize=8)
ax[0].set_xscale("log"); ax[0].set_xlabel("RND novelty ρ (log)"); ax[0].set_ylabel("atoms")
ax[0].set_title("The gate signal separates distorted (compressed OOD) from MPtrj (baseline)\n(keep tail is 10⁴× higher)", fontweight="bold", fontsize=10)
ax[0].legend(fontsize=8)
rr = np.logspace(-4, 2, 300)
ax[1].semilogx(rr, smoothstep(rr, 0.05, 0.5)**2, lw=2.6, color="#8e44ad")
ax[1].axvspan(0.05, 0.5, color="#f9e79f", alpha=.5)
ax[1].set_xlabel("RND novelty ρ (log)"); ax[1].set_ylabel("correction weight  w = smoothstep(ρ)²")
ax[1].set_title("Gate w(ρ): normal → 0 (base intact), OOD → 1 (physics)", fontweight="bold", fontsize=10)
ax[1].grid(alpha=.3); ax[1].annotate("normal\n(baseline/target)\nw≈0", (1e-3, 0.05), fontsize=8, color="#2980b9")
ax[1].annotate("OOD\n(compressed)\nw≈1", (3, 0.8), fontsize=8, color="#c0392b")
fig.suptitle("The \"WHERE\" gate: RND novelty (learned OOD signal) → correction weight w∈[0,1]", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"guide_gate_separation.png", dpi=130); plt.close(fig)
print("saved guide_gate_separation.png")

# ============ FIG C: 3 zones of V(r) ============
from pair_physics import zbl_V
r = np.linspace(0.05, 3.0, 600)
Zi, Zj = 38, 8  # Sr-O
vz = zbl_V(Zi, Zj, r)
# scheme: "true" physics = ZBL core + bonded well; "soft model" underestimates in bonded
well = -3.0*np.exp(-((r-2.2)/0.35)**2)
bonded_true = 6.0*np.exp(-((r-0.9)/0.45)**2)         # missing bonded repulsion (scheme)
true = vz + bonded_true + well
model_soft = 0.25*vz + 0.3*bonded_true + well        # model underestimates the short range
corrected = true                                      # anchor pulls up to true
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.axvspan(0.05, 0.3, color="#fdecea", alpha=.7); ax.axvspan(0.3, 1.5, color="#eafaf1", alpha=.7); ax.axvspan(1.5, 3.0, color="#f4f6f7", alpha=.7)
ax.plot(r, np.clip(model_soft, -5, 60), lw=2.4, color="#c0392b", ls="--", label="MACE (softened) — underestimates")
ax.plot(r, np.clip(corrected, -5, 60), lw=2.6, color="#27ae60", label="MACE + anchor — pulled up to physics")
ax.plot(r, np.clip(vz, -5, 60), lw=1.4, color="#7f8c8d", ls=":", label="ZBL (core, 1/r)")
ax.axhline(0, color="k", lw=.6)
ax.annotate("CORE\nZBL 1/r\n(keV–MeV,\nradiation)", (0.16, 45), ha="center", fontsize=8.5, color="#a93226", fontweight="bold")
ax.annotate("BONDED 0.3–1.5 Å\nlearned residual\nfixes softening\n(compressed structures)", (0.9, 30), ha="center", fontsize=8.5, color="#1e8449", fontweight="bold")
ax.annotate("EQUILIBRIUM >1.5 Å\ncorrection = 0\n(GNN alone, base intact)", (2.25, 20), ha="center", fontsize=8.5, color="#566573", fontweight="bold")
ax.set_xlabel("r (interatomic distance), Å"); ax.set_ylabel("pair energy V(r), eV (schematic)")
ax.set_ylim(-6, 60); ax.set_title("\"WHAT\" we add: 3 zones of one potential — anchor fixes core+bonded, stays silent at equilibrium",
                                   fontweight="bold", fontsize=11)
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIG/"guide_zones.png", dpi=130); plt.close(fig)
print("saved guide_zones.png")
