#!/usr/bin/env python3
"""T1.4 — VASP single-point FORCE reference on distorted SrTiO3 frames.

Generates ready-to-run VASP input dirs (one per frame) mirroring the vasp_tier1
convention (same INCAR head / KPOINTS-by-KSPACING / POTCAR library), so the user can
run DFT and we compare F_DFT vs F_vanilla vs F_pairphys per atom — the only test that
shows whether the anchor's short-range force change is actually *more accurate*.

Frames (SrTiO3 2x2x2 = 40 atoms, the in-domain Sr system):
  - perfect (force sanity ~0)
  - rattle σ∈{0.1,0.2,0.3}×2 seeds (mild/moderate distortion, with a min-distance floor)
  - compress V/V0∈{0.92,0.85,0.78} + small rattle (high density)
  - pushed-pair Ti-O & O-O at target contacts {1.8,1.6,1.4,1.2} Å (the anchor's active zone)

Force-appropriate INCAR deviations from the bulk-static template (documented):
  ISMEAR=0 (Gaussian — VASP recommends not -5 for forces) and LREAL=.FALSE. (accurate forces).

    python 00_make_vasp_inputs.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
TIER1_LIB = Path(os.environ.get("ZBL_VASP_TIER1_WS", "/path/to/vasp_tier1/scripts"))  # TODO: set for your machine
sys.path.insert(0, str(TIER1_LIB))
import lib.incar_templates as itmpl                                   # noqa: E402
from lib.incar_templates import kpoints_for_cell                     # noqa: E402
from lib.potcar import assemble_potcar, unique_in_order             # noqa: E402
from lib.structure_utils import isotropic_strain, min_contact_ratio  # noqa: E402

HOSTS = ROOT / "raddmg" / "hosts"
WS = ROOT / "raddmg" / "vasp_forces"
INPUTS = WS / "inputs"
FRAMES = WS / "frames"
KSPACING = 0.20


def force_static_incar(symbols, encut: int = 520) -> str:
    """vasp_tier1 static head, but ISMEAR=0 + LREAL=.FALSE. for accurate forces;
    encut bumped on short-contact (PAW-borderline) frames."""
    head = itmpl._COMMON_HEAD.replace("ENCUT  = 520", f"ENCUT  = {encut}")
    return (
        head
        + itmpl._GAMMA_SMEARING            # ISMEAR=0, SIGMA=0.05
        + "EDIFF  = 1E-7\nIBRION = -1\nNSW    = 0\nISPIN  = 2\n"
        + itmpl._magmom_line(symbols) + "\n"
        + "LREAL  = .FALSE.\nALGO   = Normal\n"
    )


def by_z(atoms):
    """Sort atoms by atomic number → contiguous element blocks (valid POSCAR/POTCAR)."""
    return atoms[np.argsort(atoms.get_atomic_numbers(), kind="stable")]


def pushed_pair(base, a_sym, b_sym, target):
    """Move a central a_sym atom along its bond to the nearest b_sym atom so their
    separation = target Å (creates one clean short contact)."""
    at = base.copy()
    center = at.cell[:].sum(0) / 2
    ai = [i for i, s in enumerate(at.get_chemical_symbols()) if s == a_sym]
    ia = ai[int(np.argmin(np.linalg.norm(at.positions[ai] - center, axis=1)))]
    bj = [i for i, s in enumerate(at.get_chemical_symbols()) if s == b_sym and i != ia]
    vecs = at.get_distances(ia, bj, mic=True, vector=True)
    k = int(np.argmin(np.linalg.norm(vecs, axis=1)))
    ib, vec = bj[k], vecs[k]
    u = vec / np.linalg.norm(vec)
    at.positions[ia] = at.positions[ib] - u * target   # place A at 'target' from B
    return at


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    base = read(str(HOSTS / "SrTiO3_conv.xyz")).repeat((2, 2, 2))    # 40 atoms

    frames = []  # (name, kind, atoms, encut)
    frames.append(("perfect", "perfect", base.copy(), 520))
    # --- warm regime: min_dist ≳1.1 Å, DFT-reliable; anchor SILENT (validates MLIP forces vs DFT) ---
    for sigma in (0.2, 0.3):
        for seed in (1, 2):
            at = base.copy(); at.rattle(stdev=sigma, seed=seed)
            frames.append((f"rattle_s{sigma:.2f}_seed{seed}", "rattle", at, 520))
    at = base.copy(); at.rattle(stdev=0.4, seed=1)
    frames.append(("rattle_s0.40_seed1", "rattle", at, 520))
    for vv in (0.90, 0.82):
        comp = isotropic_strain(base, vv - 1.0); comp.rattle(stdev=0.15, seed=7)
        frames.append((f"compress_v{vv:.2f}", "compress", comp, 520))
    for a_sym, b_sym in (("Ti", "O"), ("O", "O")):
        for target in (1.5, 1.3):
            frames.append((f"pair_{a_sym}-{b_sym}_d{target:.1f}", "pushed_pair",
                           pushed_pair(base, a_sym, b_sym, target), 520))
    # --- anchor-active probe: min_dist <1.0 Å — the ONLY regime where pairphys ≠ vanilla;
    #     PAW-borderline (sphere overlap) → ENCUT bumped, treat DFT here with caution ---
    for seed in (1, 2):
        at = base.copy(); at.rattle(stdev=0.55, seed=seed)
        frames.append((f"active_rattle_s0.55_seed{seed}", "active", at, 700))
    for a_sym, b_sym in (("Ti", "O"), ("O", "O")):
        frames.append((f"active_pair_{a_sym}-{b_sym}_d1.0", "active",
                       pushed_pair(base, a_sym, b_sym, 1.0), 700))

    manifest = []
    allframes = []
    for name, kind, at, encut in frames:
        at = by_z(at)
        d = INPUTS / name
        d.mkdir(parents=True, exist_ok=True)
        symbols = at.get_chemical_symbols()
        cell_lens = np.linalg.norm(at.cell[:], axis=1).tolist()
        write(str(d / "POSCAR"), at, format="vasp", direct=True, sort=False)
        (d / "INCAR").write_text(force_static_incar(symbols, encut))
        (d / "KPOINTS").write_text(kpoints_for_cell(cell_lens, KSPACING))
        assemble_potcar(unique_in_order(symbols), d / "POTCAR")
        ratio, _ = min_contact_ratio(at)
        dm = at.get_all_distances(mic=True)
        np.fill_diagonal(dm, np.inf)
        mind = float(dm.min())
        paw = "  PAW-borderline (<1.0 Å contact)" if kind == "active" else ""
        (d / "README.txt").write_text(
            f"SrTiO3 force-reference frame ({kind}).{paw}\nName: {name}\n"
            f"N={len(at)}  min_dist={mind:.3f} Å  min_contact_ratio={ratio:.3f}\n"
            f"Single-point: NSW=0, IBRION=-1, ISMEAR=0, LREAL=.FALSE., ENCUT={encut}\n")
        at.info["name"] = name
        at.info["kind"] = kind
        write(str(FRAMES / f"{name}.xyz"), at)
        allframes.append(at)
        manifest.append(dict(name=name, kind=kind, n_atoms=len(at), encut=encut,
                             min_dist=round(mind, 3), min_ratio=round(ratio, 3),
                             dir=str(d)))
        print(f"{name:24s} {kind:11s} N={len(at)} min_dist={mind:.2f}Å encut={encut}",
              flush=True)

    write(str(FRAMES / "all_frames.extxyz"), allframes)
    (WS / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(frames)} frames → {INPUTS}")
    print("manifest →", WS / "manifest.json")


if __name__ == "__main__":
    main()
