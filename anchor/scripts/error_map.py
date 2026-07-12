#!/usr/bin/env python3
"""Where does pairphys REMOVE force error? Per-atom map vs DFT on the keep set.

For each atom on keep_test (which has DFT reference forces F_ref):
    err_vanilla  = |F_vanilla  - F_ref|
    err_pairphys = |F_pairphys - F_ref|
    Δerr = err_vanilla - err_pairphys      (>0 → anchor removed error, <0 → overcorrected)
and bin Δerr by (a) nearest-neighbour distance, (b) element, (c) RND novelty. Uses the
existing predictions (results/all_vanilla_keep.json, results/pairphys_keep.json) for forces
+ keep_test.xyz for geometry/elements + the pairphys gate for novelty.

    python error_map.py
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

ROOT = Path(os.environ.get("ZBL_IAML_WS", "")) / "idea_uncertainty_gated_physics_anchor"  # TODO: set for your machine (external workspace)
if os.environ.get("ZBL_IAML_WS"):
    sys.path.insert(0, str(ROOT / "raddmg" / "scripts"))
from common_calc import make_calculator  # noqa: E402

PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
RES = ROOT / "results"
FIG = ROOT / "figures"
ACTIVE = 0.05          # eV/Å — atom counts as "touched by anchor" if |F_pp-F_van| > this
DIST_EDGES = [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 9.0]


def per_atom_min_dist(at) -> np.ndarray:
    i, d = neighbor_list("id", at, 6.0)
    out = np.full(len(at), np.inf)
    if len(i):
        np.minimum.at(out, i, d)
    return out


def main() -> None:
    keep = read(str(PRE / "splits" / "keep_test.xyz"), index=":")
    van = json.loads((RES / "all_vanilla_keep.json").read_text())["frames"]
    pp = json.loads((RES / "pairphys_keep.json").read_text())["frames"]
    assert len(keep) == len(van) == len(pp), (len(keep), len(van), len(pp))
    calc_pp = make_calculator("vanilla_pairphys")

    el, dmin, nov, fref, ev, ep = [], [], [], [], [], []
    for k, at in enumerate(keep):
        Fref = np.array(van[k]["F_ref"]); Fv = np.array(van[k]["F_pred"]); Fp = np.array(pp[k]["F_pred"])
        assert np.allclose(Fref, pp[k]["F_ref"], atol=1e-3), f"F_ref mismatch frame {k}"
        try:
            nv = np.asarray(calc_pp.gate.novelty(calc_pp.mace.get_descriptors(at.copy())))
        except Exception:
            nv = np.full(len(at), np.nan)
        el += at.get_chemical_symbols()
        dmin.append(per_atom_min_dist(at)); nov.append(nv)
        fref.append(np.linalg.norm(Fref, axis=1))
        ev.append(np.linalg.norm(Fv - Fref, axis=1)); ep.append(np.linalg.norm(Fp - Fref, axis=1))
    el = np.array(el)
    dmin = np.concatenate(dmin); nov = np.concatenate(nov)
    fref = np.concatenate(fref); ev = np.concatenate(ev); ep = np.concatenate(ep)
    derr = ev - ep
    active = np.abs(  # atoms the anchor actually touches
        np.where(np.isfinite(derr), derr, 0.0)) > ACTIVE

    print(f"atoms total={len(derr)}  anchor-active (|Δ|>{ACTIVE})={int(active.sum())} "
          f"({active.mean():.1%})  improved(Δerr>0)={int((derr[active]>0).sum())} "
          f"worsened={int((derr[active]<0).sum())}")
    print(f"mean Δerr over active = {derr[active].mean():+.3f} eV/Å  "
          f"(total error removed Σ = {derr.sum():+.1f} eV/Å)")

    print("\n=== by nearest-neighbour distance (active atoms) ===")
    print(f"{'r bin [Å]':>11} {'n_act':>6} {'%improv':>8} {'⟨Δerr⟩':>9} {'⟨err_van⟩':>10} {'⟨err_pp⟩':>9}")
    dbins = []
    for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:]):
        m = active & (dmin >= lo) & (dmin < hi)
        n = int(m.sum())
        row = dict(lo=lo, hi=hi, n=n,
                   pct_improved=float(np.mean(derr[m] > 0)) if n else None,
                   mean_derr=float(np.mean(derr[m])) if n else None,
                   mean_err_van=float(np.mean(ev[m])) if n else None,
                   mean_err_pp=float(np.mean(ep[m])) if n else None)
        dbins.append(row)
        if n:
            print(f"{lo:5.1f}-{hi:4.1f} {n:>6} {row['pct_improved']:>7.0%} "
                  f"{row['mean_derr']:>+9.3f} {row['mean_err_van']:>10.3f} {row['mean_err_pp']:>9.3f}")

    print("\n=== by element (active atoms) ===")
    ebins = {}
    for sp in sorted(set(el[active].tolist())):
        m = active & (el == sp)
        ebins[sp] = dict(n=int(m.sum()), mean_derr=float(np.mean(derr[m])),
                         pct_improved=float(np.mean(derr[m] > 0)))
        print(f"  {sp:3s} n={ebins[sp]['n']:4d}  ⟨Δerr⟩={ebins[sp]['mean_derr']:+.3f}  "
              f"improved={ebins[sp]['pct_improved']:.0%}")

    out = dict(active_threshold=ACTIVE, n_atoms=len(derr), n_active=int(active.sum()),
               mean_derr_active=float(derr[active].mean()), total_derr=float(derr.sum()),
               by_distance=dbins, by_element=ebins)
    (RES / "error_map.json").write_text(json.dumps(out, indent=1))
    print("\n→", RES / "error_map.json")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    a = active & np.isfinite(dmin)
    ax[0].scatter(dmin[a], derr[a], s=8, alpha=0.4, c=np.where(derr[a] > 0, "tab:green", "tab:red"))
    centers = [(b["lo"] + b["hi"]) / 2 for b in dbins if b["n"]]
    ax[0].plot(centers, [b["mean_derr"] for b in dbins if b["n"]], "k-o", lw=2, label="⟨Δerr⟩")
    ax[0].axhline(0, color="gray", lw=0.7)
    ax[0].set_xlabel("nearest-neighbour distance [Å]"); ax[0].set_ylabel("Δerr = err_van − err_pp [eV/Å]")
    ax[0].set_title("error removed vs distance (green=improved)"); ax[0].set_xlim(0.6, 2.6); ax[0].legend()

    sps = list(ebins)
    ax[1].bar(range(len(sps)), [ebins[s]["mean_derr"] for s in sps], color="tab:blue")
    ax[1].set_xticks(range(len(sps))); ax[1].set_xticklabels(sps)
    ax[1].axhline(0, color="gray", lw=0.7)
    ax[1].set_ylabel("⟨Δerr⟩ [eV/Å]"); ax[1].set_title("error removed by element (active atoms)")

    af = a & np.isfinite(nov)
    ax[2].scatter(ev[af], ep[af], s=8, alpha=0.4, c=np.where(derr[af] > 0, "tab:green", "tab:red"))
    mx = float(np.nanpercentile(ev[af], 99)) if af.any() else 1.0
    ax[2].plot([0, mx], [0, mx], "k--", lw=0.8)
    ax[2].set_xlabel("err_vanilla [eV/Å]"); ax[2].set_ylabel("err_pairphys [eV/Å]")
    ax[2].set_title("below diagonal = anchor better"); ax[2].set_xlim(0, mx); ax[2].set_ylim(0, mx)
    fig.tight_layout()
    fig.savefig(FIG / "error_map.png", dpi=140)
    print("→", FIG / "error_map.png")


if __name__ == "__main__":
    main()
