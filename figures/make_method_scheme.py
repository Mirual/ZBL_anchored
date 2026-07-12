#!/usr/bin/env python3
"""Redraw the poster's 'Method' scheme as a compact README-friendly figure.

The original poster crop is ~3.7:1 (tiny text at GitHub's ~830 px column).
This redraws the same diagram at ~1.9:1 with larger relative type.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Ellipse

# ---- palette (poster) -------------------------------------------------------
TEAL = "#17706b"
TEAL_DK = "#0e4f4b"
GREY_BG = "#f4f4f2"
GREY_BD = "#c9c9c4"
TEAL_BG = "#e9f2f0"
TEAL_BD = "#9cc4bf"
BOX_BD = "#5a5a5a"
INK = "#1c1c1c"
SUB = "#444444"
ORANGE = "#b96a12"
ORANGE_BG = "#fdf6ec"

MONO = {"family": "DejaVu Sans Mono"}

fig, ax = plt.subplots(figsize=(13.0, 6.9), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)


def band(x, y, w, h, fc, ec, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                       fc=fc, ec=ec, lw=1.4, ls=ls, mutation_aspect=100/56)
    ax.add_patch(p); return p


def box(cx, cy, w, h, title, lines=(), title_fs=13.5, fs=10.5, ec=BOX_BD,
        title_c=INK, mono_idx=(), mono_c=TEAL):
    band(cx - w/2, cy - h/2, w, h, "white", ec)
    n = len(lines)
    ty = cy + (h*0.22 if n else 0)
    ax.text(cx, ty, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color=title_c)
    for i, ln in enumerate(lines):
        yy = ty - h*0.30 - i*h*0.24
        kw = dict(ha="center", va="center", fontsize=fs, color=SUB)
        if i in mono_idx:
            kw.update(fontdict=dict(MONO), color=mono_c, fontsize=fs+0.5)
        ax.text(cx, yy, ln, **kw)


def arrow(x1, y1, x2, y2, c=BOX_BD, lw=2.0):
    ax.annotate("", (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=16))

# ---- bands ------------------------------------------------------------------
band(1.2, 60.5, 97.6, 37.5, GREY_BG, GREY_BD)
ax.text(3.4, 93.4, "F R O Z E N   F O U N D A T I O N   M L I P   —   W E I G H T S   U N C H A N G E D",
        fontsize=10.5, fontweight="bold", color="#8a8a85", va="center")

band(1.2, 12.5, 97.6, 44.5, TEAL_BG, TEAL_BD)
ax.text(3.4, 52.6, "O U R   A D D - O N", fontsize=10.5, fontweight="bold",
        color=TEAL, va="center")

# ---- top row (frozen) ---------------------------------------------------------
ty = 76.0; th = 17.0
box(10.5, ty, 15.0, th, "Structure", ["atoms, cell"])
box(33.0, ty, 24.0, th, "Universal MLIP", ["MACE · DPA-3.1 · M3GNet · CHGNet"], fs=10.0)
box(56.5, ty, 17.0, th, "E_GNN, F_GNN", ["vanilla output"])
box(88.0, ty, 18.5, th, "ZBL-anchored", ["E, F;  w = 0 → vanilla"], fs=10.0)

# plus-circle
PC = (71.5, ty)
ax.add_patch(Ellipse(PC, 3.6, 3.6 * (13.0/6.9), fc="white", ec=TEAL, lw=2.2))
ax.text(*PC, "+", ha="center", va="center", fontsize=19, color=TEAL,
        fontweight="bold")

arrow(18.2, ty, 20.8, ty)
arrow(45.2, ty, 47.8, ty)
arrow(65.2, ty, 69.4, ty)
arrow(74.3, ty, 78.5, ty)

# ---- add-on row ---------------------------------------------------------------
by = 33.5; bh = 19.5
box(27.5, by, 25.0, bh, "1 · Uncertainty ρᵢ (RND)",
    ["ρᵢ = ‖pred(dᵢ) − target(dᵢ)‖²", "small in-distribution, large OOD"],
    mono_idx=(0,), fs=9.5)
box(48.5, by, 12.5, bh, "2 · Gate w(ρ)", [" ", " "], title_fs=12.5)
box(67.0, by, 21.0, bh, "3 · Pair physics",
    ["w·[V_ZBL − V_dimer]₊", "missing repulsion from dimer scans"],
    mono_idx=(0,), fs=9.5)

# mini sigmoid inside gate box
gx = np.linspace(44.6, 52.4, 60)
gy = by - 7.0 + 7.6 / (1 + np.exp(-(gx - 48.5) * 1.9))
ax.plot(gx, gy, color=TEAL, lw=2.4, solid_capstyle="round")
ax.plot([44.6, 52.4], [by - 7.0, by - 7.0], color="#bbbbbb", lw=0.9)
ax.text(44.1, by - 6.6, "0", fontsize=8.5, color=SUB, ha="right")
ax.text(52.9, by + 0.9, "1", fontsize=8.5, color=SUB, ha="left")

# by-construction (dashed)
bc = band(89.0 - 7.9, by - 12.0, 15.8, 26.0, "#f1f7f6", TEAL, ls=(0, (5, 3)))
ax.text(89.0, by + 10.0, "by construction", ha="center", fontsize=11.5,
        fontweight="bold", color=TEAL)
ax.text(89.0, by + 4.6, "w = 0 (in-dist.):", ha="center", fontsize=9.5, color=SUB)
ax.text(89.0, by + 0.8, "bit-identical", ha="center", fontsize=10,
        color=TEAL_DK, fontdict=dict(MONO), fontweight="bold")
ax.text(89.0, by - 4.4, "w → 1 (OOD):", ha="center", fontsize=9.5, color=SUB)
ax.text(89.0, by - 8.2, "+ physics ΔV", ha="center", fontsize=10,
        color=TEAL_DK, fontdict=dict(MONO), fontweight="bold")

# descriptors arrow: Universal MLIP -> box 1
arrow(33.0, ty - th/2 - 1.0, 33.0, by + bh/2 + 1.2, c=TEAL)
ax.text(34.3, 55.0, "descriptors dᵢ", fontsize=10.5, color=TEAL,
        fontdict=dict(MONO), va="center")

# flow arrows in add-on row
arrow(40.2, by, 42.0, by, c=TEAL)
arrow(54.9, by, 56.3, by, c=TEAL)

# pair physics -> plus circle
arrow(71.5, by + bh/2 + 1.2, 71.5, ty - 3.6, c=TEAL)
ax.text(72.8, 56.5, "Σ w·ΔV", fontsize=10.5, color=TEAL,
        fontdict=dict(MONO), va="center")

# ---- calibration bar ----------------------------------------------------------
band(1.2, 2.0, 97.6, 8.0, ORANGE_BG, ORANGE)
ax.text(50.0, 6.0, "4 · SelectiveNet calibration:  θ = (r_lo, λ)  —  minimise F-MAE(OOD)  s.t.  baseline error ≤ ε",
        ha="center", va="center", fontsize=10.5, color=ORANGE,
        fontdict=dict(MONO), fontweight="bold")

out = sys.argv[1] if len(sys.argv) > 1 else "method_scheme.png"
fig.savefig(out, dpi=200, facecolor="white")
print("saved", out)
