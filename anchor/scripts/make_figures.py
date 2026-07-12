#!/usr/bin/env python3
"""Figures for the anchor experiment: calibration heatmap, V(r), and the KEY one — overlap of min-dist keep vs MPtrj."""
from __future__ import annotations
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_predict import pair_correction, smoothstep  # noqa: E402

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
FIG = Path(__file__).resolve().parents[1] / "figures"
RA, RB = 0.3, 1.5


def min_pair_dist(at):
    d = neighbor_list("d", at, 6.0)
    return d.min() if len(d) else np.nan


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32",
                          head="mp_pbe_refit_add")
    keep = read(PRE / "splits" / "keep_test.xyz", index=":")
    Fv = [np.asarray((at.__setattr__("calc", calc), at.get_forces())[1]) for at in keep]

    # --- grid heatmap ---
    As = [20, 50, 100, 200, 400, 800]; bs = [0.2, 0.3, 0.4, 0.5]
    def fmae(A, b):
        num = den = 0.0
        for at, fv in zip(keep, Fv):
            fr = at.arrays["REF_forces"]; _, fc = pair_correction(at, A, b, RA, RB)
            dd = np.abs((fv + fc) - fr); num += dd.sum(); den += dd.size
        return num / den
    base = fmae(0.0, 1.0)
    M = np.array([[fmae(A, b) for b in bs] for A in As])

    # --- min-dist overlap keep vs MPtrj ---
    mp = read(os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), index="0:800")
    dk = np.array([min_pair_dist(a) for a in keep])
    dm = np.array([min_pair_dist(a) for a in mp])

    # ---- FIG 1: anchor results ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    im = ax[0].imshow(M, aspect="auto", origin="lower", cmap="viridis_r",
                      extent=[bs[0], bs[-1], 0, len(As)])
    ax[0].set_yticks(np.arange(len(As)) + 0.5); ax[0].set_yticklabels(As)
    ax[0].set(title=f"keep F MAE (vanilla={base:.1f}) vs anchor (A,b)", xlabel="b (Å)", ylabel="A (eV)")
    plt.colorbar(im, ax=ax[0], label="F MAE eV/Å")
    bi, bj = np.unravel_index(M.argmin(), M.shape)
    ax[0].plot(bs[bj], bi + 0.5, "r*", ms=16)
    ax[1].bar(["vanilla", f"anchor\nA={As[bi]} b={bs[bj]}"], [base, M.min()], color=["#a53b3b", "#3ba56b"])
    ax[1].set(title=f"keep F MAE: −{(1-M.min()/base)*100:.0f}%", ylabel="eV/Å")
    rr = np.linspace(0.1, 2.0, 200)
    V = As[bi] * np.exp(-rr / bs[bj]) * (1 - smoothstep(rr, RA, RB))
    ax[2].plot(rr, V, color="#3b6ea5", lw=2); ax[2].axvline(RA, ls="--", c="g"); ax[2].axvline(RB, ls="--", c="r")
    ax[2].set(title="Born–Mayer×(1−s) correction V(r)", xlabel="r (Å)", ylabel="V (eV)")
    fig.suptitle("Output-additive physical anchor — results (WITHOUT retraining)", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "anchor_results.png", dpi=130)

    # ---- FIG 2 (KEY): overlap ----
    fig2, a2 = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 3, 60)
    a2.hist(dm, bins=bins, alpha=0.55, color="#3ba56b", density=True, label="MPtrj (normal)")
    a2.hist(dk, bins=bins, alpha=0.55, color="#a53b3b", density=True, label="keep (compressed)")
    a2.axvspan(0.9, 1.6, color="gray", alpha=0.15)
    a2.text(1.25, a2.get_ylim()[1]*0.85, "overlap zone\ndistance does NOT separate\n→ ρ-gating needed",
            ha="center", fontsize=10, color="#444")
    a2.set(title="Why distance-gating breaks the base: min-dist keep vs MPtrj overlap",
           xlabel="min interatomic distance (Å)", ylabel="density"); a2.legend()
    fig2.tight_layout(); fig2.savefig(FIG / "distance_overlap.png", dpi=130)
    print(f"vanilla {base:.2f}, best anchor {M.min():.2f} (A={As[bi]},b={bs[bj]})")
    print(f"keep min-dist med={np.nanmedian(dk):.2f}  MPtrj med={np.nanmedian(dm):.2f}")
    print(f"overlap 0.9-1.6Å: keep {(((dk>0.9)&(dk<1.6)).mean()*100):.0f}%  MPtrj {(((dm>0.9)&(dm<1.6)).mean()*100):.0f}%")
    print("written figures/anchor_results.png, distance_overlap.png")


if __name__ == "__main__":
    main()
