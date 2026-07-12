#!/usr/bin/env python3
"""T0.4 — force-activation map (MLIP-only, no DFT).

The anchor is a force-calibrated correction: it should change FORCES in the distorted /
short-range regime and vanish near equilibrium. The energy-based tests (T0.1/T0.2) are
blind to this by construction. Here we evaluate BOTH calculators on IDENTICAL distorted
configurations and measure the per-atom force change

    dF = |F_pairphys - F_vanilla|   [eV/Å]

versus each atom's nearest-neighbour distance and its RND novelty. This proves *where*
and *how much* the anchor engages in the radiation regime (necessary condition for a
force improvement; "better" still needs the DFT test T1.4).

Probes (all on the fluorite host):
  - static distorted cells: rattle (sigma 0.1-0.5 Å) + isotropic compression (V/V0 0.95-0.80)
  - recoil snapshots: a vanilla-driven recoil trajectory (genuine cascade close-approaches)

    python t0_force_activation.py --host UO2
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read
from ase.neighborlist import neighbor_list

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common_calc import make_calculator  # noqa: E402

HOSTS_DIR = ROOT / "raddmg" / "hosts"
RESULTS_DIR = ROOT / "raddmg" / "results"
FIG_DIR = ROOT / "raddmg" / "figures"

DIST_EDGES = [0.0, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 3.0, 1e9]
SAFE, DT_MAX, DT_MIN, CRASH_D = 0.05, 1.0 * units.fs, 1e-6 * units.fs, 0.02


def per_atom_min_dist(at) -> np.ndarray:
    i, d = neighbor_list("id", at, 4.0)
    out = np.full(len(at), np.inf)
    if len(i):
        np.minimum.at(out, i, d)
    return out


def forces(calc, at) -> np.ndarray:
    a = at.copy()
    a.calc = calc
    return np.asarray(a.get_forces())


def novelty(calc_pp, at) -> np.ndarray:
    try:
        return np.asarray(calc_pp.gate.novelty(calc_pp.mace.get_descriptors(at.copy())))
    except Exception:
        return np.full(len(at), np.nan)


def gmin(at) -> float:
    d = neighbor_list("d", at, 3.0)
    return float(d.min()) if len(d) else 9.0


def recoil_snapshots(calc, atoms0, pka, direction, energy_eV,
                     t_max_fs=300.0, maxsteps=8000, every=40):
    """Vanilla-driven recoil; collect snapshots for force comparison on identical configs."""
    at = atoms0.copy()
    at.calc = calc
    u = np.asarray(direction, float)
    u /= np.linalg.norm(u)
    v = np.zeros((len(at), 3))
    v[pka] = u
    at.set_velocities(v)
    at.set_velocities(v * np.sqrt(energy_eV / at.get_kinetic_energy()))
    m = at.get_masses()[:, None]
    v = at.get_velocities()
    f = at.get_forces()
    t = 0.0
    snaps = []
    for step in range(maxsteps):
        a = f / m
        vmax = float(np.linalg.norm(v, axis=1).max())
        dt = float(np.clip(SAFE * gmin(at) / (vmax + 1e-12), DT_MIN, DT_MAX))
        at.set_positions(at.get_positions() + v * dt + 0.5 * a * dt ** 2)
        try:
            f2 = at.get_forces()
        except Exception:
            break
        v = v + 0.5 * (a + f2 / m) * dt
        at.set_velocities(v)
        f = f2
        t += dt
        if step % every == 0:
            snaps.append(at.copy())
        if gmin(at) < CRASH_D or not np.isfinite(at.get_potential_energy()):
            break
        if t > t_max_fs * units.fs:
            break
    return snaps


def build_static(host_sup):
    cfgs = []
    for sigma in (0.1, 0.2, 0.3, 0.4, 0.5):
        for seed in (1, 2):
            at = host_sup.copy()
            at.rattle(stdev=sigma, seed=seed)
            cfgs.append((f"rattle{sigma:.1f}", at))
    for vv in (0.95, 0.90, 0.85, 0.80):
        at = host_sup.copy()
        at.set_cell(host_sup.cell[:] * vv ** (1 / 3), scale_atoms=True)
        cfgs.append((f"compress{vv:.2f}", at))
    return cfgs


def central_atom(atoms, species):
    center = atoms.cell[:].sum(axis=0) / 2.0
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == species]
    d = np.linalg.norm(atoms.get_positions()[idx] - center, axis=1)
    return idx[int(np.argmin(d))]


def binned(dist, dF, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        n = int(m.sum())
        rows.append(dict(lo=lo, hi=(hi if hi < 1e8 else None), n=n,
                         dF_mean=float(np.mean(dF[m])) if n else None,
                         dF_p90=float(np.percentile(dF[m], 90)) if n else None))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="UO2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--nrep-static", type=int, default=2)
    p.add_argument("--nrep-recoil", type=int, default=3)
    p.add_argument("--recoil-energies", nargs="+", type=float, default=[150, 300])
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    conv = read(str(HOSTS_DIR / f"{args.host}_conv.xyz"))
    cation = next(s for s in conv.get_chemical_symbols() if s != "O")

    calc_van = make_calculator("vanilla", device=args.device)
    calc_pp = make_calculator("vanilla_pairphys", device=args.device)

    # build identical configurations
    configs = build_static(conv.repeat((args.nrep_static,) * 3))
    super3 = conv.repeat((args.nrep_recoil,) * 3)
    for sp in (cation, "O"):
        pka = central_atom(super3, sp)
        for E in args.recoil_energies:
            try:
                snaps = recoil_snapshots(calc_van, super3, pka, (1, 0, 0), E)
                configs += [(f"recoil_{sp}_{int(E)}", s) for s in snaps]
            except Exception as e:
                print(f"[warn] recoil {sp} {E} failed: {e}", flush=True)

    pools = {"static": dict(dist=[], dF=[], nov=[], Fv=[], Fp=[]),
             "recoil": dict(dist=[], dF=[], nov=[], Fv=[], Fp=[])}
    for source, at in configs:
        key = "recoil" if source.startswith("recoil") else "static"
        Fv = forces(calc_van, at)
        Fp = forces(calc_pp, at)
        dF = np.linalg.norm(Fp - Fv, axis=1)
        pools[key]["dist"].append(per_atom_min_dist(at))
        pools[key]["dF"].append(dF)
        pools[key]["nov"].append(novelty(calc_pp, at))
        pools[key]["Fv"].append(np.linalg.norm(Fv, axis=1))
        pools[key]["Fp"].append(np.linalg.norm(Fp, axis=1))
    for k in pools:
        for q in pools[k]:
            pools[k][q] = (np.concatenate(pools[k][q]) if pools[k][q]
                           else np.array([]))

    summary = {"test": "t0_force_activation", "host": args.host, "by_source": {}}
    for k, P in pools.items():
        if not len(P["dist"]):
            continue
        finite = np.isfinite(P["dist"])
        summary["by_source"][k] = dict(
            n_atoms=int(len(P["dF"])),
            dF_max=float(np.max(P["dF"])) if len(P["dF"]) else None,
            frac_active=float(np.mean(P["dF"] > 0.1)),
            dF_vs_dist=binned(P["dist"][finite], P["dF"][finite], DIST_EDGES),
        )
        print(f"[{k}] atoms={len(P['dF'])} dF_max={np.max(P['dF']):.2f} eV/Å "
              f"frac(dF>0.1)={np.mean(P['dF'] > 0.1):.2f}", flush=True)
        for r in summary["by_source"][k]["dF_vs_dist"]:
            if r["n"]:
                print(f"   r∈[{r['lo']:.1f},{r['hi']}) n={r['n']:5d} "
                      f"⟨dF⟩={r['dF_mean']:.3f} p90={r['dF_p90']:.3f}", flush=True)

    out = RESULTS_DIR / f"t0_force_activation_{args.host}.json"
    out.write_text(json.dumps(summary, indent=1))
    print("→", out, flush=True)

    # figure: dF vs distance and dF vs novelty (downsampled scatter + binned line)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    rng = np.random.default_rng(0)
    for k, color in (("static", "tab:blue"), ("recoil", "tab:red")):
        P = pools[k]
        if not len(P["dist"]):
            continue
        m = np.isfinite(P["dist"]) & np.isfinite(P["dF"])
        d, y, nv = P["dist"][m], P["dF"][m], P["nov"][m]
        s = rng.choice(len(d), size=min(2000, len(d)), replace=False)
        axes[0].scatter(d[s], y[s], s=5, alpha=0.3, color=color, label=k)
        mn = np.isfinite(nv)
        ns = rng.choice(int(mn.sum()), size=min(2000, int(mn.sum())), replace=False) if mn.sum() else []
        if len(ns):
            axes[1].scatter(nv[mn][ns], y[mn][ns], s=5, alpha=0.3, color=color, label=k)
    axes[0].set_xlabel("min interatomic distance [Å]")
    axes[0].set_ylabel("|F_pairphys − F_vanilla| [eV/Å]")
    axes[0].set_title(f"{args.host} — anchor force activation vs distance")
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("RND novelty")
    axes[1].set_ylabel("|ΔF| [eV/Å]")
    axes[1].set_title("activation vs novelty (gate)")
    axes[1].set_yscale("symlog", linthresh=0.1)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"t0_force_activation_{args.host}.png", dpi=140)
    plt.close(fig)
    print("→", FIG_DIR / f"t0_force_activation_{args.host}.png", flush=True)


if __name__ == "__main__":
    main()
