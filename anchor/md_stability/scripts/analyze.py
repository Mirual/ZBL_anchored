#!/usr/bin/env python3
"""Analysis of MD runs: d_min(t) overlay of 3 calculators + E-drift (NVE) + summary table + figures."""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"
COL = {"vanilla": "#c0392b", "bornmayer": "#2980b9", "pairphys": "#27ae60"}
MODES = ["vanilla", "bornmayer", "pairphys"]


def load(prefix):
    """{(system,ensemble): {mode: data}}"""
    groups = {}
    for f in glob.glob(str(R / f"{prefix}_*.json")):
        d = json.loads(Path(f).read_text())
        key = (d["system"], d["ensemble"], d["T"])
        groups.setdefault(key, {})[d["mode"]] = d
    return groups


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "stress"
    groups = load(prefix)
    if not groups:
        print(f"no results {prefix}_*"); return
    keys = sorted(groups)
    ncol = min(3, len(keys)); nrow = (len(keys) + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
    print(f"{'system':16s} {'ens':4s} {'T':>5s} {'mode':10s} | {'dmin_min':>8s} {'survive':>7s} {'Tmean':>6s} {'Edrift/at':>9s}")
    print("-" * 80)
    for gi, key in enumerate(keys):
        a = ax[gi // ncol][gi % ncol]
        sysn, ens, T = key
        for m in MODES:
            d = groups[key].get(m)
            if not d:
                continue
            st = np.array(d["log"]["step"]); dm = np.array(d["log"]["dmin"])
            a.plot(st, dm, color=COL[m], lw=1.8, label=m,
                   ls="--" if m == "vanilla" else "-", alpha=0.9)
            if d["collapsed"]:
                a.scatter([d["survival_steps"]], [d["dmin_min"]], color=COL[m], marker="x", s=70, zorder=5)
            ed = d.get("E_drift_per_atom")
            print(f"{sysn:16s} {ens:4s} {T:5.0f} {m:10s} | {d['dmin_min']:8.3f} "
                  f"{'COLL@%d'%d['survival_steps'] if d['collapsed'] else 'OK':>7s} "
                  f"{d['T_mean']:6.0f} {('%.4f'%ed) if ed is not None else '-':>9s}")
        a.axhline(0.5, color="k", ls=":", lw=1, alpha=.5)
        a.set_title(f"{sysn} · {ens.upper()} {T:.0f}K", fontsize=9, fontweight="bold")
        a.set_xlabel("MD step (fs)"); a.set_ylabel("d_min, Å"); a.legend(fontsize=7); a.grid(alpha=.3)
        print()
    for gi in range(len(keys), nrow * ncol):
        ax[gi // ncol][gi % ncol].axis("off")
    fig.suptitle(f"MD d_min(t): vanilla vs anchor (bornmayer/pairphys)  [{prefix}]", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / f"md_{prefix}_dmin.png", dpi=130)
    print(f"saved figures/md_{prefix}_dmin.png")


if __name__ == "__main__":
    main()
