#!/usr/bin/env python3
"""After VASP runs: compare DFT forces to MLIP forces (vanilla vs pairphys) per atom.

Reads OUTCAR from each inputs/<name>/ (finished runs only), aligns with the MLIP forces
from 01 (same atom order), and reports force error vs DFT overall and binned by each
atom's nearest-neighbour distance — the headline being whether pairphys reduces the
short-range force error relative to vanilla.

    python 02_compare_forces.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
WS = ROOT / "raddmg" / "vasp_forces"
INPUTS = WS / "inputs"
FRAMES = WS / "frames"
RES = WS / "results"

DIST_EDGES = [0.0, 1.2, 1.5, 1.8, 2.1, 2.5, 99]


def per_atom_min_dist(at):
    i, d = neighbor_list("id", at, 5.0)
    out = np.full(len(at), np.inf)
    if len(i):
        np.minimum.at(out, i, d)
    return out


def read_dft_forces(name):
    for fn in ("OUTCAR", "vasprun.xml"):
        f = INPUTS / name / fn
        if f.exists() and f.stat().st_size > 0:
            try:
                return np.asarray(read(str(f), index=-1).get_forces())
            except Exception:
                continue
    return None


def fmae(a, b):
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    mlip = json.loads((RES / "mlip_forces.json").read_text())
    manifest = json.loads((WS / "manifest.json").read_text())

    rows, pooled = [], {"dist": [], "dft": [], "van": [], "pp": []}
    for m in manifest:
        name = m["name"]
        Fdft = read_dft_forces(name)
        if Fdft is None or name not in mlip:
            continue
        Fv = np.array(mlip[name]["F_vanilla"])
        Fp = np.array(mlip[name]["F_pairphys"])
        if Fdft.shape != Fv.shape:
            print(f"  SKIP {name}: shape {Fdft.shape} vs {Fv.shape}")
            continue
        at = read(str(FRAMES / f"{name}.xyz"))
        mind = per_atom_min_dist(at)
        rows.append(dict(name=name, kind=m["kind"], min_dist=m["min_dist"],
                         Fmae_vanilla=fmae(Fdft, Fv), Fmae_pairphys=fmae(Fdft, Fp)))
        pooled["dist"].append(mind)
        pooled["dft"].append(np.linalg.norm(Fdft, axis=1))
        pooled["van"].append(np.linalg.norm(Fdft - Fv, axis=1))
        pooled["pp"].append(np.linalg.norm(Fdft - Fp, axis=1))

    if not rows:
        print("No finished VASP runs found yet (need OUTCAR in inputs/<name>/).")
        return

    for k in pooled:
        pooled[k] = np.concatenate(pooled[k])

    print(f"{'frame':22s} {'kind':11s} {'min_d':>6} {'Fmae van':>9} {'Fmae pp':>9}  better")
    for r in sorted(rows, key=lambda x: x["min_dist"]):
        better = "pairphys" if r["Fmae_pairphys"] < r["Fmae_vanilla"] else "vanilla"
        print(f"{r['name']:22s} {r['kind']:11s} {r['min_dist']:6.2f} "
              f"{r['Fmae_vanilla']:9.3f} {r['Fmae_pairphys']:9.3f}  {better}")

    # error binned by per-atom nearest-neighbour distance (short-range = the anchor zone)
    print("\nper-atom force error |F_DFT - F_model| [eV/Å] by distance bin:")
    print(f"{'r bin [Å]':>12} {'n':>6} {'⟨dF⟩ van':>9} {'⟨dF⟩ pp':>9} {'⟨|F_DFT|⟩':>10}")
    bins = []
    for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:]):
        msk = (pooled["dist"] >= lo) & (pooled["dist"] < hi)
        n = int(msk.sum())
        b = dict(lo=lo, hi=(hi if hi < 90 else None), n=n,
                 dF_van=float(np.mean(pooled["van"][msk])) if n else None,
                 dF_pp=float(np.mean(pooled["pp"][msk])) if n else None,
                 F_dft=float(np.mean(pooled["dft"][msk])) if n else None)
        bins.append(b)
        if n:
            print(f"{lo:5.1f}-{hi:4.1f} {n:>11} {b['dF_van']:9.3f} {b['dF_pp']:9.3f} "
                  f"{b['F_dft']:10.3f}")

    summary = dict(per_frame=rows, by_dist=bins,
                   overall_Fmae_vanilla=float(np.mean(pooled["van"])),
                   overall_Fmae_pairphys=float(np.mean(pooled["pp"])),
                   n_frames=len(rows))
    (RES / "forces_compare.json").write_text(json.dumps(summary, indent=1))
    print("\noverall F-MAE vs DFT:  vanilla=%.3f  pairphys=%.3f eV/Å"
          % (summary["overall_Fmae_vanilla"], summary["overall_Fmae_pairphys"]))
    print("→", RES / "forces_compare.json")

    # figure: dF vs distance (van vs pp) + per-frame F-MAE bars
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    centers = [(b["lo"] + (b["hi"] or b["lo"] + 1)) / 2 for b in bins if b["n"]]
    ax[0].plot(centers, [b["dF_van"] for b in bins if b["n"]], "o-", label="vanilla")
    ax[0].plot(centers, [b["dF_pp"] for b in bins if b["n"]], "s-", label="pairphys")
    ax[0].set_xlabel("nearest-neighbour distance [Å]")
    ax[0].set_ylabel("⟨|F_DFT − F_model|⟩ [eV/Å]")
    ax[0].set_title("force error vs DFT by distance")
    ax[0].legend()
    rs = sorted(rows, key=lambda x: x["min_dist"])
    x = np.arange(len(rs))
    ax[1].bar(x - 0.2, [r["Fmae_vanilla"] for r in rs], 0.4, label="vanilla")
    ax[1].bar(x + 0.2, [r["Fmae_pairphys"] for r in rs], 0.4, label="pairphys")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r["name"] for r in rs], rotation=90, fontsize=6)
    ax[1].set_ylabel("frame F-MAE vs DFT [eV/Å]")
    ax[1].set_title("per-frame force error")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(RES / "forces_compare.png", dpi=140)
    print("→", RES / "forces_compare.png")


if __name__ == "__main__":
    main()
