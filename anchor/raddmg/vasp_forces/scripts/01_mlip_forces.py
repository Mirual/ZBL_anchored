#!/usr/bin/env python3
"""Compute MLIP forces (vanilla + vanilla_pairphys) on the SAME frames the VASP
reference uses, in the same atom order, so per-atom comparison aligns with DFT.

    python 01_mlip_forces.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from common_calc import make_calculator  # noqa: E402

WS = ROOT / "raddmg" / "vasp_forces"
FRAMES = WS / "frames"
RES = WS / "results"


def forces(calc, at) -> np.ndarray:
    a = at.copy()
    a.calc = calc
    return np.asarray(a.get_forces())


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    frames = read(str(FRAMES / "all_frames.extxyz"), index=":")
    cv = make_calculator("vanilla")
    cp = make_calculator("vanilla_pairphys")
    out = {}
    for at in frames:
        name = at.info.get("name", at.get_chemical_formula())
        out[name] = dict(
            symbols=at.get_chemical_symbols(),
            F_vanilla=forces(cv, at).tolist(),
            F_pairphys=forces(cp, at).tolist(),
        )
        dF = np.linalg.norm(np.array(out[name]["F_pairphys"]) -
                            np.array(out[name]["F_vanilla"]), axis=1)
        print(f"{name:22s} max|F_pp-F_van|={dF.max():.3f} eV/Å", flush=True)
    (RES / "mlip_forces.json").write_text(json.dumps(out, indent=1))
    print("→", RES / "mlip_forces.json")


if __name__ == "__main__":
    main()
