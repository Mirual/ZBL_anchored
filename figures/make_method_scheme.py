#!/usr/bin/env python3
"""Redraw the poster's 'Method' scheme as a compact README-friendly figure.

The original poster crop was ~3.7:1 (tiny text at GitHub's ~830 px column).
This draws the same diagram at ~1.5:1 with large type: at equal column width
the text renders roughly at document size (12-17 px).
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

FW, FH = 11.5, 7.6  # inches; ~1.5:1
fig, ax = plt.subplots(figsize=(FW, FH), dpi=220)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)


def band(x, y, w, h, fc, ec, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.2",
                       fc=fc, ec=ec, lw=1.6, ls=ls, mutation_aspect=100/56)
    ax.add_patch(p); return p


def box(cx, cy, w, h, title, lines=(), title_fs=17, fs=12, ec=BOX_BD,
        title_c=INK, mono_idx=(), mono_c=TEAL):
    band(cx - w/2, cy - h/2, w, h, "white", ec)
    n = len(lines)
    ty = cy + (h*0.24 if n else 0)
    ax.text(cx, ty, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color=title_c)
    for i, ln in enumerate(lines):
        yy = ty - h*0.30 - i*h*0.24
        kw = dict(ha="center", va="center", fontsize=fs, color=SUB)
        if i in mono_idx:
            kw.update(fontdict=dict(MONO), color=mono_c, fontsize=fs+0.5)
        ax.text(cx, yy, ln, **kw)


def arrow(x1, y1, x2, y2, c=BOX_BD, lw=2.4):
    ax.annotate("", (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=19))

# ---- bands ------------------------------------------------------------------
band(1.2, 62.0, 97.6, 36.0, GREY_BG, GREY_BD)
ax.text(3.2, 94.6, "FROZEN FOUNDATION MLIP — WEIGHTS UNCHANGED",
        fontsize=13, fontweight="bold", color="#8a8a85", va="center")

band(1.2, 11.5, 97.6, 47.0, TEAL_BG, TEAL_BD)
ax.text(3.2, 54.8, "OUR ADD-ON", fontsize=13, fontweight="bold",
        color=TEAL, va="center")

# ---- top row (frozen) ---------------------------------------------------------
ty = 77.5; th = 22.0
box(9.3, ty, 13.6, th, "Structure", ["atoms, cell"], title_fs=16)
box(30.8, ty, 23.2, th, "Universal MLIP",
    ["MACE · DPA-3.1", "M3GNet · CHGNet"], title_fs=16, fs=11.5)
box(53.2, ty, 15.2, th, "E_GNN, F_GNN", ["vanilla output"], title_fs=14.5, fs=11.5)
box(84.3, ty, 24.6, th, "ZBL-anchored", ["E, F;  w = 0 → vanilla"],
    title_fs=16, fs=11.5)

# plus-circle (true circle: compensate axis anisotropy)
PC = (67.0, ty)
ax.add_patch(Ellipse(PC, 4.4, 4.4 * (FW/FH), fc="white", ec=TEAL, lw=2.4))
ax.text(PC[0], PC[1] - 0.4, "+", ha="center", va="center", fontsize=24,
        color=TEAL, fontweight="bold")

arrow(16.8, ty, 18.9, ty)
arrow(43.1, ty, 45.2, ty)
arrow(61.5, ty, 64.7, ty)
arrow(69.3, ty, 71.5, ty)

# ---- add-on row ---------------------------------------------------------------
by = 32.0; bh = 26.0
box(23.0, by, 30.0, bh, "1 · Uncertainty ρᵢ (RND)",
    ["ρᵢ = ‖pred(dᵢ) − target(dᵢ)‖²", "small in-distribution, large OOD"],
    title_fs=15.5, mono_idx=(0,), fs=11)
box(47.6, by, 12.6, bh, "2 · Gate w(ρ)", [" ", " "], title_fs=14)
box(67.0, by, 21.0, bh, "3 · Pair physics",
    ["w·[V_ZBL − V_dimer]₊", "missing repulsion from dimer scans"],
    title_fs=15.5, mono_idx=(0,), fs=9.3)

# mini sigmoid inside the gate box (lower half)
gx = np.linspace(43.6, 51.6, 60)
gy = by - 9.4 + 9.6 / (1 + np.exp(-(gx - 47.6) * 1.7))
ax.plot(gx, gy, color=TEAL, lw=2.8, solid_capstyle="round")
ax.plot([43.6, 51.6], [by - 9.4, by - 9.4], color="#bbbbbb", lw=1.0)
ax.text(43.1, by - 9.0, "0", fontsize=10.5, color=SUB, ha="right")
ax.text(52.1, by + 0.6, "1", fontsize=10.5, color=SUB, ha="left")

# by-construction (dashed)
band(89.5 - 8.25, by - 13.0, 16.5, 27.5, "#f1f7f6", TEAL, ls=(0, (5, 3)))
ax.text(89.5, by + 10.6, "by construction", ha="center", fontsize=13.5,
        fontweight="bold", color=TEAL)
ax.text(89.5, by + 5.2, "w = 0 (in-dist.):", ha="center", fontsize=11, color=SUB)
ax.text(89.5, by + 1.0, "bit-identical", ha="center", fontsize=11.5,
        color=TEAL_DK, fontdict=dict(MONO), fontweight="bold")
ax.text(89.5, by - 4.8, "w → 1 (OOD):", ha="center", fontsize=11, color=SUB)
ax.text(89.5, by - 9.0, "+ physics ΔV", ha="center", fontsize=11.5,
        color=TEAL_DK, fontdict=dict(MONO), fontweight="bold")

# descriptors arrow: Universal MLIP -> box 1
arrow(30.8, ty - th/2 - 1.2, 30.8, by + bh/2 + 1.4, c=TEAL)
ax.text(32.4, 56.0, "descriptors dᵢ", fontsize=12.5, color=TEAL,
        fontdict=dict(MONO), va="center")

# flow arrows in the add-on row
arrow(38.7, by, 40.8, by, c=TEAL)
arrow(54.4, by, 56.0, by, c=TEAL)

# pair physics -> plus circle
arrow(67.0, by + bh/2 + 1.4, 67.0, ty - 4.9, c=TEAL)
ax.text(68.4, 57.5, "Σ w·ΔV", fontsize=12.5, color=TEAL,
        fontdict=dict(MONO), va="center")

# ---- calibration bar ----------------------------------------------------------
band(1.2, 1.5, 97.6, 8.0, ORANGE_BG, ORANGE)
ax.text(50.0, 5.5, "4 · SelectiveNet calibration:  θ = (r_lo, λ) — minimise F-MAE(OOD)  s.t.  baseline error ≤ ε",
        ha="center", va="center", fontsize=12.5, color=ORANGE,
        fontdict=dict(MONO), fontweight="bold")

out = sys.argv[1] if len(sys.argv) > 1 else "method_scheme.png"
fig.savefig(out, dpi=220, facecolor="white")
print("saved", out)
