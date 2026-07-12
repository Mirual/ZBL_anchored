#!/usr/bin/env python3
"""Energy + forces prediction with a MACE foundation on an extxyz file.

Writes results/predictions_<tag>.json:
    {"model": tag, "model_path": str, "data": str,
     "frames": [{"idx", "n_atoms", "symbols": [...], "atomic_numbers": [...],
                 "E_ref", "E_pred",
                 "F_ref": [[fx,fy,fz], ...] or None,
                 "F_pred": [[fx,fy,fz], ...]}, ...]}

F_ref is emitted only when the dataset has reference forces on the ASE Atoms
(`REF_forces` array, falling back to `forces`); F_pred is always populated
from `atoms.get_forces()`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
from tqdm import tqdm

WS = Path(__file__).resolve().parents[1]

MODEL_FILES = {
    "mh0": WS / "pretrained" / "mace-mh-0.model",
    "mh1": WS / "pretrained" / "mace-mh-1.model",
    "mh0_ft_eprio": WS / "results" / "mace_mh0_zbl_finetune" / "mace_mh0_zbl_finetune.model",
}


def get_ref_energy(atoms) -> float:
    info = atoms.info
    if "REF_energy" in info:
        return float(info["REF_energy"])
    if "energy" in info:
        return float(info["energy"])
    return float(atoms.get_potential_energy())


def get_ref_forces(atoms) -> np.ndarray | None:
    """Return reference forces array (n_atoms, 3) or None if unavailable.

    Prefers `REF_forces` (the convention in 26_02_5and8 and the MPtrj samples);
    falls back to `forces` to support datasets that only set the standard key.
    """
    arrays = atoms.arrays
    for key in ("REF_forces", "forces"):
        if key in arrays:
            return np.asarray(arrays[key], dtype=float)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_FILES), required=True)
    p.add_argument("--head", default="mp_pbe_refit_add",
                   help="Multi-head MACE-MH selector. Dataset is MP-PBE, so default = mp_pbe_refit_add.")
    p.add_argument("--data", default=str(WS / "data" / "test.xyz"))
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--skip-on-error", action="store_true",
                   help="Skip frames whose MACE prediction raises (e.g. out-of-vocab atomic number "
                        "for restricted z-table models). Record null E_pred/F_pred + error string.")
    args = p.parse_args()

    model_path = MODEL_FILES[args.model]
    tag = f"{args.model}_{args.head}"
    out_path = Path(args.out) if args.out else WS / "results" / f"predictions_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_path}  ({model_path.stat().st_size/1e6:.1f} MB) on {args.device} / {args.dtype}  head={args.head}")
    t0 = time.time()
    calc = MACECalculator(
        model_paths=[str(model_path)], device=args.device,
        default_dtype=args.dtype, head=args.head,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"reading {args.data}")
    frames = read(args.data, index=":", format="extxyz")
    print(f"  {len(frames)} frames")

    records = []
    t0 = time.time()
    n_skipped = 0
    for i, atoms in enumerate(tqdm(frames, desc=f"predict[{args.model}]")):
        e_ref = get_ref_energy(atoms)
        f_ref = get_ref_forces(atoms)
        atoms.calc = calc
        try:
            e_pred: float | None = float(atoms.get_potential_energy())
            f_pred_arr = np.asarray(atoms.get_forces(), dtype=float)
            f_pred_out = f_pred_arr.tolist()
            err: str | None = None
        except Exception as exc:
            if not args.skip_on_error:
                raise
            n_skipped += 1
            e_pred = None
            f_pred_out = None
            err = f"{type(exc).__name__}: {exc}"
            print(f"  [skip frame {i}] {err}")
        records.append({
            "idx": i,
            "n_atoms": len(atoms),
            "symbols": list(atoms.get_chemical_symbols()),
            "atomic_numbers": [int(z) for z in atoms.get_atomic_numbers()],
            "E_ref": e_ref,
            "E_pred": e_pred,
            "F_ref": f_ref.tolist() if f_ref is not None else None,
            "F_pred": f_pred_out,
            "error": err,
        })

    wall = time.time() - t0
    print(f"  done in {wall:.1f}s  ({wall/len(frames):.2f}s/frame); skipped {n_skipped}/{len(frames)}")

    payload = {
        "model": args.model,
        "head": args.head,
        "tag": tag,
        "model_path": str(model_path),
        "data": str(args.data),
        "device": args.device,
        "dtype": args.dtype,
        "wall_s": wall,
        "frames": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
