#!/usr/bin/env python3
"""Dimer-scan plots: log-log V(r) (ZBL divergence vs anchor plateau) + r_min(E) (ceiling of each variant)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"
PAIRS = ["Sr-Sr", "Sr-O"]
STYLE = {
    "vanilla_zblon":    ("vanilla ZBL-on", "#111111", "-", 2.4),
    "vanilla_zbloff":   ("vanilla ZBL-off (GNN)", "#c0392b", "-", 1.8),
    "bornmayer_zbloff": ("bornmayer ZBL-off", "#2980b9", "-", 1.8),
    "pairphys_zbloff":  ("pairphys ZBL-off", "#27ae60", "-", 1.8),
    "pairphys_core_zbloff": ("pairphys+core-ZBL (fix)", "#8e44ad", "-", 2.2),
}
ELAB = {"200": "200 eV", "2000": "2 keV", "20000": "20 keV", "200000": "200 keV", "2000000": "2 MeV"}


def arr(x):
    return np.array([v if v is not None else np.nan for v in x], float)


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for row, pair in enumerate(PAIRS):
        d = json.loads((RES / f"scan_{pair}.json").read_text())
        r = np.array(d["r"]); aV = ax[row][0]; aR = ax[row][1]
        # --- panel V(r) ---
        for tag, (lab, c, ls, lw) in STYLE.items():
            V = arr(d["V"][tag])
            aV.loglog(r, np.where(V > 0, V, np.nan), ls, color=c, lw=lw, label=lab)
        aV.loglog(r, arr(d["V_zbl_analytic"]), "--", color="#7f8c8d", lw=1.3, label="ZBL (analytic)")
        aV.loglog(r, arr(d["V_coulomb"]), ":", color="#999999", lw=1.1, label="Coulomb Z·Z/r")
        for e, el in ELAB.items():
            aV.axhline(float(e), color="k", lw=.5, alpha=.25)
            aV.annotate(el, (r.min() * 1.1, float(e) * 1.15), fontsize=6.5, color="#555")
        aV.set_title(f"{pair}: pair energy V(r)", fontweight="bold")
        aV.set_xlabel("r, Å"); aV.set_ylabel("V(r), eV"); aV.set_ylim(1, 2e7)
        aV.legend(fontsize=7, loc="upper right"); aV.grid(alpha=.25, which="both")
        # --- panel r_min(E) ---
        Es = np.array([float(e) for e in d["rmin"]["vanilla_zblon"]])
        for tag, (lab, c, ls, lw) in STYLE.items():
            rm = arr([d["rmin"][tag][f"{e:.0f}"] for e in Es])
            aR.loglog(Es, rm, "o-", color=c, lw=lw, ms=5, label=lab)
        rmz = arr([d["rmin"]["zbl_analytic"][f"{e:.0f}"] for e in Es])
        aR.loglog(Es, rmz, "s--", color="#7f8c8d", lw=1.3, ms=4, label="ZBL (analytic)")
        # mark the "passes through" zone
        aR.axvspan(3.4e4, Es.max() * 1.5, color="#f9e0e0", alpha=.5)
        aR.annotate("ZBL-off PASS THROUGH\n(GNN ceiling ~34 keV)", (1.2e5, 0.5), fontsize=7.5, color="#a93226")
        aR.set_title(f"{pair}: turning point r_min(E)", fontweight="bold")
        aR.set_xlabel("collision energy, eV"); aR.set_ylabel("r_min, Å")
        aR.legend(fontsize=7, loc="lower left"); aR.grid(alpha=.25, which="both")
    fig.suptitle("Sr at keV–MeV: ZBL and pairphys+core-ZBL (fix) diverge as Z·Z/r → physical turning point;\n"
                 "bare anchor variants are bounded (ceiling ~34 keV = GNN extrapolation) → projectile passes through",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "dimer_scan_Sr.png", dpi=130)
    print("saved figures/dimer_scan_Sr.png")


if __name__ == "__main__":
    main()
