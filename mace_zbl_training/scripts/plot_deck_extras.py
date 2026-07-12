#!/usr/bin/env python3
"""Generate supplementary plots for STATUS_DECK:
   - parity_overlay.png       zero-shot vs fine-tuned on the same axes
   - residual_hist.png        log-y histogram of shifted residuals (eV/atom)
   - per_element_shift.png    bar chart of FT vs ZS per-element shift
   - element_distribution.png test-set element count histogram
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WS = Path(__file__).resolve().parents[1]
RESULTS = WS / "results"

sys.path.insert(0, str(WS.parent / "innp" / "finetune"))
from importlib import import_module
mm = import_module("06_compare_metrics")

ZS = json.load(open(RESULTS / "predictions_mh0_mp_pbe_refit_add.json"))["frames"]
FT = json.load(open(RESULTS / "predictions_mh0_ft_eprio_Default.json"))["frames"]
M = json.load(open(RESULTS / "metrics.json"))


def shifted_pred(frames):
    Eref = np.array([f["E_ref"] for f in frames])
    Epred = np.array([f["E_pred"] for f in frames])
    n = np.array([f["n_atoms"] for f in frames])
    counts, elems = mm.build_count_matrix(frames)
    c, Esh = mm.fit_element_shift(Eref, Epred, counts)
    return Eref, Esh, n, c, elems


def parity_overlay():
    Eref_zs, Esh_zs, n_zs, _, _ = shifted_pred(ZS)
    Eref_ft, Esh_ft, n_ft, _, _ = shifted_pred(FT)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, (Eref, Esh, label, color) in zip(axes, [
        (Eref_zs, Esh_zs, "Zero-shot MACE-MH-0 (+ZBL)", "#dc2626"),
        (Eref_ft, Esh_ft, "Fine-tuned MACE-MH-0 (e-prio 10:1)", "#16a34a"),
    ]):
        ax.scatter(Eref, Esh, s=14, alpha=0.7, color=color)
        lo = float(min(Eref.min(), Esh.min()))
        hi = float(max(Eref.max(), Esh.max()))
        pad = 0.02 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("E_ref (DFT) [eV]")
        ax.set_ylabel("E_pred (shift-corrected) [eV]")
        rmse = float(np.sqrt(np.mean((Eref - Esh) ** 2)))
        mae = float(np.mean(np.abs(Eref - Esh)))
        n_atoms = np.array([f["n_atoms"] for f in (ZS if label.startswith("Zero") else FT)])
        rmse_a = float(np.sqrt(np.mean(((Eref - Esh) / n_atoms) ** 2)))
        mae_a = float(np.mean(np.abs((Eref - Esh) / n_atoms)))
        ax.set_title(f"{label}\nMAE = {mae_a:.1f} eV/atom · RMSE = {rmse_a:.1f} eV/atom")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
    fig.suptitle("Zero-shot vs fine-tuned · per-element-shifted parity · 174 test frames", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "parity_overlay.png", dpi=140)
    plt.close(fig)


def residual_hist():
    Eref_zs, Esh_zs, n_zs, _, _ = shifted_pred(ZS)
    Eref_ft, Esh_ft, n_ft, _, _ = shifted_pred(FT)
    res_zs = (Eref_zs - Esh_zs) / n_zs
    res_ft = (Eref_ft - Esh_ft) / n_ft
    bins = np.linspace(-200, 200, 81)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.hist(res_zs, bins=bins, alpha=0.55, color="#dc2626", label=f"Zero-shot (median |res| = {np.median(np.abs(res_zs)):.1f} eV/atom)")
    ax.hist(res_ft, bins=bins, alpha=0.65, color="#16a34a", label=f"Fine-tuned (median |res| = {np.median(np.abs(res_ft)):.1f} eV/atom)")
    ax.set_yscale("log")
    ax.set_xlabel("E_ref − E_pred (shift-corrected) [eV/atom]")
    ax.set_ylabel("count (log)")
    ax.set_title("Residual distribution per frame · 174 test frames")
    ax.axvline(0, color="black", lw=0.6, alpha=0.7)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "residual_hist.png", dpi=140)
    plt.close(fig)


def per_element_shift_bar():
    ft_shifts = M["mh0_ft_eprio_Default"]["shift_per_element_eV"]
    zs_shifts = M["mh0_mp_pbe_refit_add"]["shift_per_element_eV"]
    elems = sorted(ft_shifts.keys() & zs_shifts.keys(), key=lambda e: zs_shifts[e])
    x = np.arange(len(elems))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.4
    ax.bar(x - w/2, [zs_shifts[e] for e in elems], w, color="#dc2626", alpha=0.8, label="Zero-shot shift (large = reference offset)")
    ax.bar(x + w/2, [ft_shifts[e] for e in elems], w, color="#16a34a", alpha=0.8, label="Fine-tuned shift (small = E0s absorbed)")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(elems)
    ax.set_xlabel("element")
    ax.set_ylabel("per-element shift [eV/atom]")
    ax.set_title("Per-element LSQ shift on E_ref − E_pred · the FT model absorbs the offset")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "per_element_shift.png", dpi=140)
    plt.close(fig)


def element_distribution():
    z_to_sym = {3:"Li",7:"N",8:"O",9:"F",12:"Mg",15:"P",16:"S",17:"Cl",22:"Ti",
                25:"Mn",26:"Fe",38:"Sr",39:"Y",40:"Zr",43:"Tc",55:"Cs",90:"Th",
                92:"U",93:"Np"}
    counts = {}
    for fr in FT:
        for z in fr["atomic_numbers"]:
            sym = z_to_sym.get(int(z), f"Z{int(z)}")
            counts[sym] = counts.get(sym, 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    syms = [k for k, _ in items]
    cnts = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bars = ax.bar(syms, cnts, color="#0284c7", alpha=0.85)
    for b, c in zip(bars, cnts):
        ax.text(b.get_x() + b.get_width()/2, c + max(cnts)*0.012, f"{c:,}",
                ha="center", va="bottom", fontsize=9, color="#1a1a1a")
    ax.set_ylabel("atom count across 174 test frames")
    ax.set_xlabel("element")
    ax.set_title(f"Test-set element distribution · {len(syms)} of 19 elements present · {sum(cnts):,} atoms total")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "element_distribution.png", dpi=140)
    plt.close(fig)


def main():
    parity_overlay()
    residual_hist()
    per_element_shift_bar()
    element_distribution()
    print("wrote:")
    for p in ("parity_overlay", "residual_hist", "per_element_shift", "element_distribution"):
        f = RESULTS / f"{p}.png"
        print(f"  {f}  ({f.stat().st_size/1024:.1f} kB)")


if __name__ == "__main__":
    main()
