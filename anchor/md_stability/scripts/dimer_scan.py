#!/usr/bin/env python3
"""Static pair-energy scan V(r) for 4 potentials at ultra-short r (nuclear-stopping regime).
Answers "how do the 4 variants behave at Sr, 2 MeV" WITHOUT MD (which is unresolvable at 2 MeV:
v~300 Å/fs ≫ step). V(r) = potential behavior; turning point r_min(E)=min r with V(r)≥E = result
of a head-on collision with relative KE=E.

4 variants: vanilla(ZBL-on) | vanilla(ZBL-off) | bornmayer(ZBL-off) | pairphys(ZBL-off)
+ analytic true ZBL (zbl_V) + pure Coulomb as references.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as AN
from mace.calculators import MACECalculator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))   # idea/scripts for pair_physics
from anchor_calculator import AnchorCalculator          # noqa: E402
from pair_physics import zbl_V                           # noqa: E402

K_COUL = 14.3996   # eV·Å
ENERGIES = [200.0, 2e3, 2e4, 2e5, 2e6]   # eV: 200 eV … 2 MeV


def dimer(Zi, Zj, r):
    return Atoms(numbers=[Zi, Zj], positions=[[0, 0, 0], [r, 0, 0]], cell=[30, 30, 30], pbc=False)


def Vcurve(calc, Zi, Zj, rs, eref):
    V = []
    for r in rs:
        at = dimer(Zi, Zj, r); at.calc = calc
        try:
            e = float(at.get_potential_energy())
            V.append(e - eref if np.isfinite(e) else None)
        except Exception:
            V.append(None)
    return V


def find_rmin(rs, V, E):
    """min r with V(r)≥E (turning point). None → 'pass-through' (potential ceiling < E)."""
    r = np.array(rs, float); v = np.array([x if x is not None else np.nan for x in V], float)
    m = np.isfinite(v) & (v > 0)
    r, v = r[m], v[m]
    if len(r) < 2:
        return None
    o = np.argsort(r); r, v = r[o], v[o]
    if v[0] < E:                 # even at minimal r the wall is below E → passes through
        return None
    idx = np.where(v < E)[0]
    if len(idx) == 0:
        return float(r[-1])      # stops beyond the grid (weak collision)
    i = idx[0]
    r0, r1, v0, v1 = r[i - 1], r[i], v[i - 1], v[i]
    lr = np.log(r0) + (np.log(E) - np.log(v0)) * (np.log(r1) - np.log(r0)) / (np.log(v1) - np.log(v0))
    return float(np.exp(lr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="Sr-Sr")
    p.add_argument("--rmin", type=float, default=0.01)
    p.add_argument("--rmax", type=float, default=2.0)
    p.add_argument("--n", type=int, default=70)
    p.add_argument("--rref", type=float, default=6.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    a, b = args.pair.split("-"); Zi, Zj = AN[a], AN[b]
    rs = np.logspace(np.log10(args.rmin), np.log10(args.rmax), args.n).tolist()

    modes = [("vanilla_zblon", "vanilla", False, False),
             ("vanilla_zbloff", "vanilla", True, False),
             ("bornmayer_zbloff", "bornmayer", True, False),
             ("pairphys_zbloff", "pairphys", True, False),
             ("pairphys_core_zbloff", "pairphys", True, True)]   # core_zbl: analytic ZBL below r_floor
    curves = {}
    for tag, mode, dz, cz in modes:
        calc = AnchorCalculator(mode=mode, device="cuda", disable_zbl=dz, core_zbl=cz)
        at = dimer(Zi, Zj, args.rref); at.calc = calc; eref = float(at.get_potential_energy())
        curves[tag] = Vcurve(calc, Zi, Zj, rs, eref)
        print(f"{tag:18s}: V(r) ready (eref={eref:.2f})")

    # analytic references
    rar = np.array(rs)
    v_zbl = (zbl_V(Zi, Zj, rar)).tolist()
    v_coul = (K_COUL * Zi * Zj / rar).tolist()

    # turning points
    rmin_table = {}
    for tag in list(curves) + ["zbl_analytic", "coulomb"]:
        Vc = curves.get(tag) or (v_zbl if tag == "zbl_analytic" else v_coul)
        rmin_table[tag] = {f"{E:.0f}": find_rmin(rs, Vc, E) for E in ENERGIES}

    out = dict(pair=args.pair, Zi=Zi, Zj=Zj, r=rs, V=curves,
               V_zbl_analytic=v_zbl, V_coulomb=v_coul, energies=ENERGIES, rmin=rmin_table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n→ {args.out}")
    print(f"\nturning point r_min(Å) at 2 MeV:")
    for tag in ["vanilla_zblon", "vanilla_zbloff", "bornmayer_zbloff", "pairphys_zbloff",
                "pairphys_core_zbloff", "zbl_analytic"]:
        rm = rmin_table[tag]["2000000"]
        print(f"  {tag:18s}: {('%.4f'%rm) if rm else 'PASSES THROUGH (ceiling < 2 MeV)'}")


if __name__ == "__main__":
    main()
