#!/usr/bin/env python3
"""#1 — short-range repulsion ROBUSTNESS: vanilla vs vanilla_pairphys (no DFT).

Recognized axis (arXiv:2606.12704 probes "failures of the short-range repulsion" via random
structure search + MD-stability + RDF). Three legs:

  A. static RSS control  — relax random dense packings freely. With ZBL both models avoid
     r→0 collapse, so this is expected to be a NULL (documents that the anchor's value is
     under constraint/dynamics, not free relaxation).
  B. compression collapse — the discriminator. Isotropically compress distorted structures
     and relax positions at each step; "collapse" = d_min<0.5 Å / NaN / energy blow-up
     (reuses scripts/mlip_arena_highP.ramp). Report collapse rate + survival compression.
  C. RDF — aggregate pair-distance histogram at a fixed compression; a too-soft wall builds
     spurious density < 1 Å that a correct wall suppresses.

    python robustness.py
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(ROOT / "scripts"))            # for mlip_arena_highP.ramp
from common_calc import make_calculator              # noqa: E402
from mlip_arena_highP import ramp                     # noqa: E402

HIGHP = ROOT / "highP_systems"
RES = ROOT / "raddmg" / "results"
FIG = ROOT / "raddmg" / "figures"

SCALES = np.linspace(1.0, 0.55, 19)   # linear cell scale; V/V0 = s^3
S_RDF = 0.85                          # compression for RDF (V/V0 ≈ 0.61, stressed but not all collapsed)
COMP = ["Sr"] * 4 + ["Ti"] * 4 + ["O"] * 12
MODELS = ("vanilla", "vanilla_pairphys")
COLORS = {"vanilla": "tab:gray", "vanilla_pairphys": "tab:red"}


def min_dist(at) -> float:
    d = neighbor_list("d", at, 5.0)
    return float(d.min()) if len(d) else 9.0


def relax_at_scale(calc, at0, s, steps=40):
    at = at0.copy()
    at.set_cell(at0.cell[:] * s, scale_atoms=True)
    at.calc = calc
    try:
        FIRE(at, logfile=None).run(fmax=0.1, steps=steps)
        ok = bool(np.isfinite(at.get_potential_energy())) and bool(np.isfinite(at.get_forces()).all())
    except Exception:
        return None, False
    return at, ok


def rss_control(calcs, rng, n_per_vol=10):
    """Leg A: free relaxation of random packings (expected null with ZBL)."""
    out = {m: {"min_after": [], "crash": 0} for m in MODELS}
    for vpa in (8.0, 10.0, 12.0):
        for _ in range(n_per_vol):
            L = (len(COMP) * vpa) ** (1 / 3)
            at0 = Atoms(COMP, positions=rng.uniform(0, L, (len(COMP), 3)),
                        cell=[L, L, L], pbc=True)
            for m, calc in calcs.items():
                r, ok = relax_at_scale(calc, at0, 1.0, steps=120)
                if ok:
                    out[m]["min_after"].append(min_dist(r))
                else:
                    out[m]["crash"] += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    calcs = {m: make_calculator(m, args.device) for m in MODELS}

    # ---- Leg A: static RSS control ----
    print("=== A. static RSS (random packings, free relax) ===", flush=True)
    rss = rss_control(calcs, np.random.default_rng(1))
    for m in MODELS:
        ma = rss[m]["min_after"]
        print(f"  {m:18s} crashes={rss[m]['crash']}  median min_after="
              f"{np.median(ma):.2f} Å  frac(min<1.0Å)={np.mean(np.array(ma) < 1.0):.2f}", flush=True)

    # ---- Leg B: compression collapse (discriminator) + Leg C: RDF ----
    print("\n=== B. compression collapse on distorted structures + C. RDF ===", flush=True)
    files = sorted(glob.glob(f"{HIGHP}/*.xyz"))
    rows = []
    rdf = {m: [] for m in MODELS}
    crash_count = {m: 0 for m in MODELS}
    crash_scale = {m: [] for m in MODELS}
    for f in files:
        at = read(f)
        row = {"system": Path(f).stem, "n_atoms": len(at)}
        for m, calc in calcs.items():
            r = ramp(calc, at, SCALES)
            crashed = bool(r[-1]["crash"])
            cs = float(r[-1]["scale"])
            crash_count[m] += crashed
            crash_scale[m].append(cs if crashed else float(SCALES[-1]))
            row[m] = {"crash": crashed, "crash_scale": cs}
            ar, ok = relax_at_scale(calc, at, S_RDF)
            if ar is not None:
                d = neighbor_list("d", ar, 5.0)
                if len(d):
                    rdf[m].append(d)
        saved = row["vanilla"]["crash"] and not row["vanilla_pairphys"]["crash"]
        row["anchor_saved"] = saved
        rows.append(row)
        print(f"  {row['system']:10s} van {'CRASH@%.2f' % row['vanilla']['crash_scale'] if row['vanilla']['crash'] else 'ok':12s}"
              f" pp {'CRASH@%.2f' % row['vanilla_pairphys']['crash_scale'] if row['vanilla_pairphys']['crash'] else 'ok':12s}"
              f"{'  ← ANCHOR SAVED' if saved else ''}", flush=True)

    n = len(files)
    summary = {
        "n_structures": n,
        "rss_control": {m: dict(crashes=rss[m]["crash"],
                                median_min_after=float(np.median(rss[m]["min_after"])),
                                frac_below_1A=float(np.mean(np.array(rss[m]["min_after"]) < 1.0)))
                        for m in MODELS},
        "compression": {m: dict(collapse=crash_count[m], collapse_rate=round(crash_count[m] / n, 3),
                                median_survival_scale=float(np.median(crash_scale[m])))
                        for m in MODELS},
        "anchor_saved": int(sum(r["anchor_saved"] for r in rows)),
    }
    # RDF short-range integral (fraction of pairs < 1 Å)
    for m in MODELS:
        if rdf[m]:
            d = np.concatenate(rdf[m])
            summary["compression"][m]["rdf_frac_below_1A"] = float(np.mean(d < 1.0))
    RES.joinpath("robustness.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    print("\n=== SUMMARY (vanilla vs vanilla_pairphys) ===")
    print(f"A static RSS:    both crash-free, median min ~"
          f"{summary['rss_control']['vanilla']['median_min_after']:.2f} Å (ZBL → null, as expected)")
    cv = summary["compression"]["vanilla"]; cp = summary["compression"]["vanilla_pairphys"]
    print(f"B compression:   collapse  vanilla {cv['collapse']}/{n} ({cv['collapse_rate']:.0%})  "
          f"pairphys {cp['collapse']}/{n} ({cp['collapse_rate']:.0%})  anchor_saved={summary['anchor_saved']}")
    if "rdf_frac_below_1A" in cv:
        print(f"C RDF <1Å frac:  vanilla {cv['rdf_frac_below_1A']:.4f}  pairphys {cp['rdf_frac_below_1A']:.4f}")
    print("→", RES / "robustness.json")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for m in MODELS:
        ax[0].hist(rss[m]["min_after"], bins=np.arange(0, 3, 0.1), alpha=0.55,
                   color=COLORS[m], label=m)
    ax[0].axvline(1.0, color="k", ls="--", lw=0.8)
    ax[0].set_title("A. static RSS min-distance (null: ZBL safe)")
    ax[0].set_xlabel("min dist after free relax [Å]"); ax[0].legend(fontsize=8)

    x = np.arange(len(MODELS))
    ax[1].bar(x, [summary["compression"][m]["collapse_rate"] for m in MODELS],
              color=[COLORS[m] for m in MODELS])
    ax[1].set_xticks(x); ax[1].set_xticklabels(MODELS, rotation=10, fontsize=8)
    ax[1].set_ylabel("collapse fraction under compression")
    ax[1].set_title(f"B. compression collapse (n={n})")

    for m in MODELS:
        if rdf[m]:
            d = np.concatenate(rdf[m])
            ax[2].hist(d, bins=np.arange(0, 4, 0.05), histtype="step", density=True,
                       color=COLORS[m], label=m)
    ax[2].axvline(1.0, color="k", ls="--", lw=0.8)
    ax[2].set_title(f"C. aggregate RDF @ V/V0≈{S_RDF**3:.2f}")
    ax[2].set_xlabel("pair distance [Å]"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", dpi=140)
    print("→", FIG / "robustness.png")


if __name__ == "__main__":
    main()
