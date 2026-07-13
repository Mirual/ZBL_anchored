#!/usr/bin/env python3
"""Where the anchor acts — drawn from REAL data.

Curves come from (1) the exact analytic ZBL used by the code
(anchor/scripts/pair_physics.py) and (2) the model's real dimer table
(dimer_packed.pkl: ΔV = [V_ZBL − V_dimer]₊ · f_cut on a per-pair r grid,
floor 0.30 Å, cutoff at the pair's equilibrium bond).

Usage:  python3 make_zones_figure.py [out.png] [dimer_packed.pkl]
        (table path may also come from $ZBL_DIMER_PKL)
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- palette (poster) -------------------------------------------------------
TEAL = "#17706b"
GREY_C = "#8a8a85"
INK = "#1c1c1c"
SUB = "#444444"
RED = "#b03a2e"
GREEN = "#1e8449"
ORANGE = "#b96a12"
MONO = {"family": "DejaVu Sans Mono"}

# ---- exact ZBL (same constants as anchor/scripts/pair_physics.py) -----------
K_E = 14.399645
_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
_D = np.array([3.19980, 0.94229, 0.40290, 0.20162])


def zbl_V(Zi, Zj, r):
    a = 0.46850 / (Zi ** 0.23 + Zj ** 0.23)
    phi = (_C * np.exp(-_D * np.asarray(r)[:, None] / a)).sum(1)
    return K_E * Zi * Zj / np.asarray(r) * phi


# ---- real dimer table --------------------------------------------------------
OUT = sys.argv[1] if len(sys.argv) > 1 else "anchor_zones.png"
PKL = (sys.argv[2] if len(sys.argv) > 2
       else os.environ.get("ZBL_DIMER_PKL", "/path/to/dimer_packed.pkl"))
d = pickle.load(open(PKL, "rb"))
keys = [tuple(k) for k in d["keys"]]
lens = np.asarray(d["lens"]); off = np.concatenate([[0], np.cumsum(lens)])
R, DV, RCUT = np.asarray(d["r"]), np.asarray(d["dV"]), np.asarray(d["r_cut"])


def pair(Zi, Zj):
    i = keys.index((Zi, Zj)) if (Zi, Zj) in keys else keys.index((Zj, Zi))
    return R[off[i]:off[i + 1]], DV[off[i]:off[i + 1]], float(RCUT[i])


FLOOR = 0.30
MED_RCUT = float(np.median(RCUT))

fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=220)
fig.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.235)
XMIN, XMAX = 0.15, 3.5

# ================= main panel: O-O ============================================
Z1 = Z2 = 8
rt, dvt, rc_oo = pair(Z1, Z2)                    # table grid + real correction
nz = dvt > 0
v_zbl_t = zbl_V(Z1, Z2, rt)
v_model = v_zbl_t - dvt                          # model wall = ZBL - ΔV (real)

rr = np.linspace(XMIN, XMAX, 500)
ax.plot(rr, zbl_V(Z1, Z2, rr), color=INK, lw=3.2,
        label="analytic ZBL wall (exact)")
ax.plot(rt[nz], v_model[nz], "--o", color=GREY_C, lw=2.4, ms=5,
        label="vanilla model wall  =  ZBL − ΔV  (real table)")
ax.fill_between(rt[nz], v_model[nz], v_zbl_t[nz], color=RED, alpha=0.18)
ax.plot(rt[nz], v_zbl_t[nz], color=TEAL, lw=6.0, alpha=0.95, zorder=5,
        solid_capstyle="round", label="anchored wall (w = 1):  model + ΔV = ZBL")

# real softness annotation at the r=0.8 grid point
j = int(np.argmin(np.abs(rt - 0.8)))
ax.annotate(f"at {rt[j]:.1f} Å the model wall is\n"
            f"{v_model[j]:.0f} eV vs {v_zbl_t[j]:.0f} eV ZBL "
            f"(×{v_zbl_t[j]/v_model[j]:.1f} too soft)",
            xy=(rt[j], v_model[j]), xytext=(1.16, 0.55),
            fontsize=11.5, color=RED, ha="left",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
ax.annotate("missing repulsion ΔV\n(real dimer-table data)",
            xy=(0.62, 60), xytext=(0.26, 700),
            fontsize=11.5, color=RED, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))

ax.set_yscale("log"); ax.set_ylim(0.3, 15000); ax.set_xlim(XMIN, XMAX)
ax.set_ylabel("pair energy (eV)", fontsize=13)
ax.tick_params(labelsize=11.5)
ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.03), fontsize=11.5,
          frameon=False)

# ---- zones (data-driven boundaries) ----
ax.axvspan(XMIN, FLOOR, color="#9a9a94", alpha=0.18, lw=0)
ax.axvspan(FLOOR, rc_oo, color=RED, alpha=0.075, lw=0)
ax.axvspan(rc_oo, XMAX, color=GREEN, alpha=0.07, lw=0)
ax.set_xlabel("interatomic distance r (Å)", fontsize=13.5)
ax.text((XMIN + FLOOR) / 2, 9.0, "analytic ZBL core < 0.30 Å", fontsize=10.5,
        color="#6d6d68", ha="center", va="center", fontweight="bold", rotation=90)
ax.text((FLOOR + rc_oo) / 2 + 0.05, 8800,
        f"correction window (O–O):\n{FLOOR:.2f}–{rc_oo:.1f} Å", fontsize=12.5,
        color=RED, ha="center", va="top", fontweight="bold")
ax.text((rc_oo + XMAX) / 2, 8800,
        "beyond the equilibrium bond:\nΔV ≡ 0  →  bit-identical to vanilla",
        fontsize=12.5, color=GREEN, ha="center", va="top", fontweight="bold")
ax.text(0.985, 0.40, "O–O pair — every curve from real data", fontsize=11.5,
        color=INK, fontweight="bold", ha="right", transform=ax.transAxes)

fig.text(0.085, 0.030,
         "Real data: exact analytic ZBL (pair_physics.zbl_V) + the MACE-MH dimer table "
         "ΔV = [V_ZBL − V_dimer]₊·f_cut.\n"
         "The window is pair-specific: floor 0.30 Å, cutoff at the pair's equilibrium bond "
         "(O–O 1.2 Å, Ti–O 2.0 Å, Sr–Ti 3.2 Å; median over 3 570 pairs 2.8 Å).\n"
         "The RND gate additionally zeroes ΔV on in-distribution structures (w = 0).",
         fontsize=10.5, color=SUB, va="bottom")

fig.savefig(OUT, facecolor="white")
print("saved", OUT)
