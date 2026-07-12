#!/usr/bin/env python3
"""Benchmark a MACE model (e.g. MACE-MH-1, head omat_pbe) on a fixed set of crystals.

Runs single-point energy/forces/stress on a deterministic list of crystalline
supercells and records the results + timing to JSON. Run it once with the STOCK
mace install and once with the PATCHED ("ours") install, then diff the two JSON
files with compare_runs.py.

The structures are built deterministically and cached to an extxyz file so that
the stock and patched runs use byte-identical geometry (the only thing that may
differ between runs is the code/config, never the input).

Usage (minimal, apples-to-apples accuracy check, fp64 e3nn):
    python bench_mace.py --model /path/to/mace-mh-1.model --device cuda \
        --dtype float64 --head omat_pbe --out stock.json --label stock

Speed knobs (run on the PATCHED install to see the wins):
    --enable-cueq            # lossless ~6x on the symmetric contraction (needs cuequivariance)
    --dtype float32          # ~2x bandwidth, sub-meV deviation
    --compile-mode default   # torch.compile

Requires: torch, e3nn, ase, and a mace install (stock or patched). For MACE-MH-1
you must supply the model checkpoint yourself (ASL / non-commercial weights).
"""

import argparse
import json
import platform
import time

import numpy as np


def build_structures(seed: int = 0):
    """Deterministic crystalline supercells with a fixed rattle (nonzero forces)."""
    from ase.build import bulk

    specs = [
        ("Cu", dict(crystalstructure="fcc", a=3.61), (3, 3, 3)),
        ("Al", dict(crystalstructure="fcc", a=4.05), (3, 3, 3)),
        ("Fe", dict(crystalstructure="bcc", a=2.87), (3, 3, 3)),
        ("Si", dict(crystalstructure="diamond", a=5.43), (2, 2, 2)),
        ("NaCl", dict(crystalstructure="rocksalt", a=5.64), (2, 2, 2)),
        ("MgO", dict(crystalstructure="rocksalt", a=4.21), (2, 2, 2)),
    ]
    rng = np.random.default_rng(seed)
    atoms_list = []
    for name, kw, rep in specs:
        atoms = bulk(name, **kw).repeat(rep)
        # deterministic displacement (seeded), kept small to stay in-distribution
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        atoms.info["label"] = f"{name}_{rep[0]}{rep[1]}{rep[2]}"
        atoms_list.append(atoms)
    return atoms_list


def get_structures(path: str, seed: int):
    """Load cached structures if present, else build and cache them."""
    import os

    from ase.io import read, write

    if os.path.exists(path):
        return read(path, index=":")
    atoms_list = build_structures(seed)
    write(path, atoms_list)
    return atoms_list


def make_calc(args):
    from mace.calculators import MACECalculator

    kwargs = {}
    if args.head:
        kwargs["head"] = args.head
    return MACECalculator(
        model_paths=args.model,
        device=args.device,
        default_dtype=args.dtype,
        enable_cueq=args.enable_cueq,
        compile_mode=args.compile_mode,
        **kwargs,
    )


def sync(device: str):
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="path to MACE .model checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--head", default="omat_pbe", help="readout head to evaluate")
    p.add_argument("--enable-cueq", action="store_true")
    p.add_argument("--compile-mode", default=None, help="torch.compile mode, e.g. 'default'")
    p.add_argument("--structures", default="structures.extxyz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=20, help="timed single-point evals per structure")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="run")
    args = p.parse_args()

    atoms_list = get_structures(args.structures, args.seed)
    calc = make_calc(args)

    results = []
    for atoms in atoms_list:
        atoms.calc = calc
        # correctness: energy / forces / stress
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        stress = atoms.get_stress()  # Voigt 6-vector

        # timing: repeated single-point (energy+forces+stress) evaluations
        for _ in range(args.warmup):
            atoms.calc.calculate(atoms)
        sync(args.device)
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            atoms.calc.calculate(atoms)
        sync(args.device)
        dt = (time.perf_counter() - t0) / args.repeats

        results.append(
            {
                "label": atoms.info.get("label", atoms.get_chemical_formula()),
                "formula": atoms.get_chemical_formula(),
                "natoms": int(len(atoms)),
                "energy": energy,
                "energy_per_atom": energy / len(atoms),
                "forces": forces.tolist(),
                "stress": np.asarray(stress).tolist(),
                "sp_time_s": dt,
            }
        )
        print(f"{results[-1]['label']:>12}  E={energy:14.6f} eV  "
              f"|F|max={np.abs(forces).max():.4f} eV/A  t={dt*1e3:.1f} ms")

    import torch

    out = {
        "label": args.label,
        "config": {
            "dtype": args.dtype,
            "device": args.device,
            "head": args.head,
            "enable_cueq": args.enable_cueq,
            "compile_mode": args.compile_mode,
            "repeats": args.repeats,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "total_sp_time_s": sum(r["sp_time_s"] for r in results),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  (total single-point time {out['total_sp_time_s']*1e3:.1f} ms)")


if __name__ == "__main__":
    main()
