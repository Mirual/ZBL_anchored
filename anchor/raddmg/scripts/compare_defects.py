#!/usr/bin/env python3
"""Compare MLIP defect formation energies to DFT/empirical literature.

Compares the mu-free quantities (Frenkel pairs, antisite — atom-conserving, no chemical
potential needed) directly to published values. Isolated vacancies are reported as raw
ΔE only (mu-dependent, NOT directly comparable to literature formation energies).

    python compare_defects.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
RES = ROOT / "raddmg" / "results"
FIG = ROOT / "raddmg" / "figures"

# literature ranges [eV] for mu-free defects (lo, hi, note)
LIT = {
    "UO2": {
        "O_frenkel":      (3.0, 4.5, "DFT/empirical O Frenkel"),
        "U_frenkel":      (9.5, 19.0, "cation Frenkel (wide)"),
    },
    "SrTiO3": {
        "O_frenkel":      (8.0, 12.0, "estimated ~10"),
        "Sr_frenkel":     (15.0, 20.0, "high (estimated)"),
        "Ti_frenkel":     (16.0, 22.0, "high (estimated)"),
        "antisite_Sr_Ti": (4.0, 6.0, "Ti_Sr ~5 (HSE)"),
    },
}


def load(host, calc):
    f = RES / f"t0_defect_{host}_{calc}.json"
    return json.loads(f.read_text())["defects"] if f.exists() else None


def main() -> None:
    summary = {}
    for host in ("UO2", "SrTiO3"):
        van = load(host, "vanilla")
        pp = load(host, "vanilla_pairphys")
        if van is None:
            continue
        print(f"\n=== {host} — defect formation energy [eV] ===")
        print(f"{'defect':16s} {'van':>9} {'pairphys':>9} {'μ-free':>7}  literature")
        names = sorted(van)
        summary[host] = {}
        for n in names:
            v = van[n]["dE"]
            p = pp[n]["dE"] if pp and n in pp else None
            mf = van[n]["mu_free"]
            lit = LIT.get(host, {}).get(n)
            litstr = (f"{lit[0]}–{lit[1]}  ({lit[2]})" if lit
                      else ("(μ-dep, not comparable)" if not mf else "—"))
            summary[host][n] = dict(vanilla=v, pairphys=p, mu_free=mf,
                                    lit=(lit[:2] if lit else None))
            print(f"{n:16s} {v:>9.3f} {('%.3f'%p) if p is not None else '   -':>9} "
                  f"{str(mf):>7}  {litstr}")

    (RES / "defects_compare.json").write_text(json.dumps(summary, indent=1))
    print("\n→", RES / "defects_compare.json")

    # figure: mu-free defects, vanilla vs pairphys vs lit band, per host
    hosts = [h for h in ("UO2", "SrTiO3") if h in summary]
    if not hosts:
        return
    fig, axes = plt.subplots(1, len(hosts), figsize=(6 * len(hosts), 4), squeeze=False)
    for ax, host in zip(axes[0], hosts):
        names = [n for n in summary[host] if summary[host][n]["mu_free"]]
        x = np.arange(len(names))
        van = [summary[host][n]["vanilla"] for n in names]
        pp = [summary[host][n]["pairphys"] for n in names]
        ax.bar(x - 0.2, van, 0.4, label="vanilla")
        ax.bar(x + 0.2, [p if p is not None else 0 for p in pp], 0.4, label="pairphys")
        for i, n in enumerate(names):
            lit = summary[host][n]["lit"]
            if lit:
                ax.fill_between([i - 0.45, i + 0.45], lit[0], lit[1],
                                color="green", alpha=0.18,
                                label="lit range" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("formation energy [eV]")
        ax.set_title(f"{host} — μ-free defects vs literature")
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "defects_compare.png", dpi=140)
    plt.close(fig)
    print("→", FIG / "defects_compare.png")


if __name__ == "__main__":
    main()
