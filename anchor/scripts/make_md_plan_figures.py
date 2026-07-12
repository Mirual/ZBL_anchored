#!/usr/bin/env python3
"""Figures for the MD-stability plan: mechanism (real dimer), pipeline, expected result."""
import os, sys; from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, "scripts")
from mace.calculators import MACECalculator
from pair_physics import DimerCache, zbl_V
from ase.data import atomic_numbers as AN, covalent_radii
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")

# ===== FIG 1: softening→collapse mechanism (real Ti-O dimer) =====
calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
dc = DimerCache(calc)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for col, (a, b) in enumerate([("Ti", "O"), ("O", "O")]):
    Zi, Zj = AN[a], AN[b]; e = dc.get(Zi, Zj)
    r = np.linspace(0.4, e["r"][-1] + 0.3, 300)
    from ase import Atoms
    vm = np.array([calc.get_potential_energy(Atoms([Zi, Zj], positions=[[0,0,0],[rr,0,0]], cell=[20]*3, pbc=False)) for rr in r])
    vm -= calc.get_potential_energy(Atoms([Zi, Zj], positions=[[0,0,0],[6,0,0]], cell=[20]*3, pbc=False))
    vz = zbl_V(Zi, Zj, r)
    dV = np.interp(r, e["r"], e["dV"], left=e["dV"][0], right=0.0)
    vc = vm + dV
    rcov = covalent_radii[Zi] + covalent_radii[Zj]
    A = ax[col]
    A.plot(r, np.clip(vm, -20, 80), lw=2.4, color="#c0392b", label="MACE-MH-0 (softened)")
    A.plot(r, np.clip(vz, -20, 80), lw=1.6, ls="--", color="#7f8c8d", label="ZBL (true repulsion)")
    A.plot(r, np.clip(vc, -20, 80), lw=2.4, color="#27ae60", label="MACE + anchor (residual)")
    A.axvspan(0.4, e["r_cut"], color="#f39c12", alpha=0.12)
    A.axvline(rcov, color="#2980b9", ls=":", lw=1.3)
    A.annotate("equilibrium\nbond", (rcov, 60), color="#2980b9", fontsize=8, ha="center")
    A.annotate("collapse zone\n(softening)", ((0.4+e["r_cut"])/2, -12), color="#b9770e", fontsize=8, ha="center")
    A.set_title(f"{a}–{b}: pair energy", fontweight="bold")
    A.set_xlabel("r, Å"); A.set_ylabel("V(r), eV"); A.set_ylim(-20, 80); A.legend(fontsize=8); A.grid(alpha=.3)
fig.suptitle("Mechanism: foundation underestimates repulsion → atoms collapse in MD; anchor restores it",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"md_plan_mechanism.png", dpi=130); plt.close(fig)
print("saved md_plan_mechanism.png")

# ===== FIG 2: pipeline =====
fig, ax = plt.subplots(figsize=(13, 4)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 6)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=c, ec="k", lw=1.3))
    ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=8.5, fontweight="bold")
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#34495e"))
box(0.3, 4, 2.4, 1.4, "structure\n(perovskite)", "#ecf0f1")
box(3.2, 4, 2.6, 1.4, "AnchorCalculator\n(ASE)", "#d6eaf8")
box(3.2, 1.6, 2.6, 1.4, "MACE-MH-0\n+ RND gate\n+ correction", "#d5f5e3")
box(6.3, 4, 2.3, 1.4, "ASE MD\nNVT / NVE", "#fcf3cf")
box(9.1, 4, 2.3, 1.4, "trajectory\n+ per-step log", "#fadbd8")
box(11.9, 4, 1.8, 1.4, "metrics", "#e8daef")
box(9.1, 1.6, 4.6, 1.4, "d_min(t) · E-drift · T · survival · RDF", "#e8daef")
arr(2.7, 4.7, 3.2, 4.7); arr(5.8, 4.7, 6.3, 4.7); arr(8.6, 4.7, 9.1, 4.7); arr(11.4, 4.7, 11.9, 4.7)
arr(4.5, 4, 4.5, 3.0); arr(11.4, 3.9, 11.4, 3.0)
ax.set_title("MD-stability pipeline: 3 calculators × {compressed, normal, hot} × {NVT, NVE}",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"md_plan_pipeline.png", dpi=130); plt.close(fig)
print("saved md_plan_pipeline.png")

# ===== FIG 3: expected result (scheme) =====
t = np.linspace(0, 10, 500)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
# d_min(t): vanilla collapses, anchor holds
van = 1.8 - 1.3*np.clip((t-3)/2, 0, 1)**1.5 + 0.05*np.sin(8*t)
van[t>5] = 0.3 + 0.05*np.sin(8*t[t>5])
anc = 1.75 + 0.07*np.sin(7*t) - 0.1*np.clip((t-3)/6,0,1)
ax[0].plot(t, van, lw=2.2, color="#c0392b", label="vanilla (collapse)")
ax[0].plot(t, anc, lw=2.2, color="#27ae60", label="anchor (holds)")
ax[0].axhline(0.7, ls="--", color="k", lw=1, alpha=.6); ax[0].annotate("phys. limit", (8, 0.78), fontsize=8)
ax[0].set_title("compressed/hot: d_min(t)", fontweight="bold")
ax[0].set_xlabel("t, ps"); ax[0].set_ylabel("min interatomic dist., Å"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_ylim(0,2.2)
# normal: RDF matches
r = np.linspace(0.5, 6, 400)
g = np.exp(-((r-2.0)/0.25)**2) + 0.7*np.exp(-((r-3.4)/0.35)**2) + 0.4*np.exp(-((r-4.5)/0.5)**2)
ax[1].plot(r, g, lw=2.4, color="#34495e", label="vanilla")
ax[1].plot(r, g*(1+0.01*np.sin(20*r)), lw=1.4, ls="--", color="#27ae60", label="anchor (≡ vanilla)")
ax[1].set_title("normal: RDF g(r) — baseline preserved", fontweight="bold")
ax[1].set_xlabel("r, Å"); ax[1].set_ylabel("g(r)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle("Expected result: anchor prevents collapse in compressed cases, does not alter normal dynamics",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"md_plan_expected.png", dpi=130); plt.close(fig)
print("saved md_plan_expected.png (scheme)")
