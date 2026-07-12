#!/usr/bin/env python3
"""Tier-0 figures: EOS curves, defect formation energies, recoil onset. Each panel is
guarded so a missing/failed result never aborts the others.

    python make_figures.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
RESULTS_DIR = ROOT / "raddmg" / "results"
FIG_DIR = ROOT / "raddmg" / "figures"


def load(pattern):
    return [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob(pattern))]


def fig_eos():
    data = load("t0_eos_*.json")
    if not data:
        return
    hosts = sorted({h for d in data for h in d["hosts"]})
    fig, axes = plt.subplots(1, len(hosts), figsize=(4 * len(hosts), 3.4), squeeze=False)
    for ax, host in zip(axes[0], hosts):
        for d in data:
            r = d["hosts"].get(host)
            if not r:
                continue
            e = [x - min(r["energies"]) for x in r["energies"]]
            ax.plot(r["volumes"], e, "o-", label=d["calc"], ms=4)
        ax.set_title(host)
        ax.set_xlabel("Volume [Å³]")
        ax.set_ylabel("E − E_min [eV]")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "t0_eos.png", dpi=140)
    plt.close(fig)
    print("→ t0_eos.png")


def fig_defect():
    data = load("t0_defect_*.json")
    if not data:
        return
    by_host = {}
    for d in data:
        by_host.setdefault(d["host"], []).append(d)
    for host, ds in by_host.items():
        names = sorted({n for d in ds for n in d["defects"]})
        fig, ax = plt.subplots(figsize=(7, 3.6))
        w = 0.8 / max(len(ds), 1)
        for i, d in enumerate(ds):
            vals = [d["defects"].get(n, {}).get("dE", float("nan")) for n in names]
            xs = [j + i * w for j in range(len(names))]
            ax.bar(xs, vals, width=w, label=d["calc"])
        ax.set_xticks([j + 0.4 for j in range(len(names))])
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("ΔE [eV]")
        ax.set_title(f"{host} — defect formation (μ-free: Frenkel/antisite)")
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"t0_defect_{host}.png", dpi=140)
        plt.close(fig)
        print(f"→ t0_defect_{host}.png")


def fig_recoil():
    data = load("t0_recoil_*.json")
    if not data:
        return
    by_host = {}
    for d in data:
        by_host.setdefault(d["host"], []).append(d)
    for host, ds in by_host.items():
        species = sorted({r["species"] for d in ds for r in d["rows"] if "species" in r})
        fig, axes = plt.subplots(1, len(species), figsize=(4.2 * len(species), 3.4),
                                 squeeze=False)
        for ax, sp in zip(axes[0], species):
            for d in ds:
                rows = [r for r in d["rows"] if r.get("species") == sp and "n_frenkel" in r]
                # average Frenkel pairs over directions at each energy
                by_e = {}
                for r in rows:
                    by_e.setdefault(r["energy_eV"], []).append(r["n_frenkel"])
                xs = sorted(by_e)
                ys = [sum(by_e[x]) / len(by_e[x]) for x in xs]
                ax.plot(xs, ys, "o-", label=d["calc"], ms=4)
            ax.set_title(f"{host} — PKA {sp}")
            ax.set_xlabel("recoil energy [eV]")
            ax.set_ylabel("⟨surviving Frenkel pairs⟩")
            ax.set_xscale("log")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"t0_recoil_{host}.png", dpi=140)
        plt.close(fig)
        print(f"→ t0_recoil_{host}.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (fig_eos, fig_defect, fig_recoil):
        try:
            fn()
        except Exception as e:
            print(f"[warn] {fn.__name__} failed: {e}")


if __name__ == "__main__":
    main()
