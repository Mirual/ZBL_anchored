#!/usr/bin/env python3
"""Figures for the ρ-gated anchor explainer: ρ separation, pipeline diagram, distance-vs-ρ."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parents[1] / "figures"
RES = Path(__file__).resolve().parents[1] / "results"


def fig_rho_separation():
    z = np.load(RES / "rho_check.npz"); rk, rm = z["rk"], z["rm"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(rm, bins=bins, density=True, alpha=0.6, color="#3ba56b", label=f"MPtrj (baseline), med={np.median(rm):.2f}")
    ax.hist(rk, bins=bins, density=True, alpha=0.6, color="#a53b3b", label=f"distorted (compressed OOD), med={np.median(rk):.2f}")
    ax.axvline(0.5, ls="--", c="k", lw=0.8)
    ax.set(title="ρ SEPARATES baseline from distorted (which distance could not do)",
           xlabel="ρ = extrapolation-score (0=in-dist, 1=OOD)", ylabel="density")
    ax.legend()
    ax.text(0.05, ax.get_ylim()[1]*0.6, "MPtrj\npiles up at ρ=0\n→ correction OFF\n→ base intact", color="#2a7", fontsize=10)
    ax.text(0.62, ax.get_ylim()[1]*0.4, "keep stretches to ρ=1\n→ correction ON\n→ fixing distortion", color="#a33", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "rho_separation.png", dpi=130)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.5)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=c, ec="k", lw=1.2))
        ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=10)
    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=18, lw=1.5, color="#333"))
    box(0.3, 3, 2, 1, "structure\n(atoms, r)", "#eee")
    # upper branch — GNN (untouched)
    box(3, 5, 2.6, 1.1, "MACE-MH-0 (GNN)\n→ E_GNN, F_GNN", "#cfe")
    # lower branch — ρ
    box(3, 1.5, 2.6, 1.1, "atom descriptors\n(256-dim, ready)", "#fec")
    box(6.2, 1.5, 2.3, 1.1, "ρ = kNN-dist\nto MPtrj reference", "#fec")
    box(6.2, 0.0, 2.3, 1.0, "gate w(ρ)\n0 if confident, 1 if OOD", "#fdd")
    # physics
    box(9, 1.5, 2.3, 1.1, "physics V(r)\nBorn–Mayer repulsive", "#dfd")
    # sum
    box(9.4, 4.6, 2.6, 1.4, "E = E_GNN + Σ w(ρ)·V(r)\nF = F_GNN + Σ w(ρ)·F_phys", "#ddf")
    arr(2.3, 3.5, 3, 5.4); arr(2.3, 3.4, 3, 2.0)
    arr(5.6, 2.05, 6.2, 2.05); arr(7.3, 1.5, 7.3, 1.0)
    arr(8.5, 2.05, 9, 2.05)
    arr(5.6, 5.5, 9.4, 5.3)
    arr(10.2, 2.6, 10.5, 4.6)
    ax.text(6.5, 6.6, "ρ-gated physical anchor: correction triggers BY CONFIDENCE, not by r",
            ha="center", fontsize=13, weight="bold")
    ax.text(10.7, 3.7, "weight w(ρ)\ncontrols\ncorrection strength", fontsize=9, color="#449")
    fig.savefig(FIG / "pipeline.png", dpi=130, bbox_inches="tight")


def fig_distance_vs_rho():
    z = np.load(RES / "rho_check.npz"); rk, rm = z["rk"], z["rm"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    # emulation: distance does not separate (taken conceptually from overlap data)
    ax[0].set_title("Distance: normal and distorted OVERLAP ✗")
    ax[0].text(0.5, 0.5, "min-dist keep ≈ MPtrj\nin the 0.9–1.6 Å range\n(see distance_overlap.png)\n→ distance-gate hits the base",
               ha="center", va="center", transform=ax[0].transAxes, fontsize=12, color="#a33")
    ax[0].axis("off")
    bins = np.linspace(0, 1, 40)
    ax[1].hist(rm, bins=bins, density=True, alpha=0.6, color="#3ba56b", label="MPtrj")
    ax[1].hist(rk, bins=bins, density=True, alpha=0.6, color="#a53b3b", label="keep")
    ax[1].set(title="ρ: SEPARATES ✓", xlabel="ρ", ylabel="density"); ax[1].legend()
    fig.suptitle("Why ρ works where distance does not", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "distance_vs_rho.png", dpi=130)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_rho_separation(); fig_pipeline(); fig_distance_vs_rho()
    print("written: rho_separation.png, pipeline.png, distance_vs_rho.png")


if __name__ == "__main__":
    main()
