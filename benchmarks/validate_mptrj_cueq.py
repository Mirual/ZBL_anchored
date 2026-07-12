#!/usr/bin/env python3
"""Validate that enable_cueq is lossless (and time it) vs e3nn on REAL MPtraj structures.

Runs MACE-MH-1 single-point with cueq OFF (e3nn) and ON (fused cueq) on a sample
of MPtraj frames (diverse chemistry/sizes) and reports per-structure energy/force
agreement AND speedup. The check is OFF-vs-ON correctness (must match to round-off).
DFT REF_forces (reference-free) give a bonus sanity MAE. Note: MPtraj cells are
small (~4-144 atoms) so cueq speedup is modest here — see the size-binned summary.

    python validate_mptrj_cueq.py --structures mptrj.xyz --model mace.model
"""
import argparse
import json
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structures", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--head", default="omat_pbe")
    p.add_argument("--n", type=int, default=0, help="limit number of frames (0 = all)")
    p.add_argument("--dtypes", default="float64,float32")
    p.add_argument("--repeats", type=int, default=10, help="timed single-points per structure")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", default=None, help="optional JSON dump of all rows")
    args = p.parse_args()

    import torch
    from ase.io import read
    from mace.calculators import MACECalculator

    frames = read(args.structures, index=":")
    if args.n:
        frames = frames[: args.n]
    print(f"loaded {len(frames)} MPtraj frames from {args.structures}")
    print(f"GPU {torch.cuda.get_device_name(0)} | "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    def make(dtype, enable):
        return MACECalculator(model_paths=args.model, device="cuda",
                              default_dtype=dtype, head=args.head, enable_cueq=enable)

    def timed(calc, src):
        a = src.copy(); a.calc = calc
        for _ in range(args.warmup):
            calc.calculate(a)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            calc.calculate(a)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.repeats

    all_rows = {}
    for dtype in args.dtypes.split(","):
        print("\n" + "=" * 104)
        print(f"dtype = {dtype}   (cueq OFF=e3nn  vs  ON=fused cueq)")
        print("=" * 104)
        c_off, c_on = make(dtype, False), make(dtype, True)
        rows = []
        for i, src in enumerate(frames):
            n = len(src)
            try:
                a = src.copy(); a.calc = c_off
                e_off, f_off = float(a.get_potential_energy()), a.get_forces()
                a = src.copy(); a.calc = c_on
                e_on, f_on = float(a.get_potential_energy()), a.get_forces()
                t_off, t_on = timed(c_off, src), timed(c_on, src)
                ref_f = src.arrays.get("REF_forces")
                fmae = (float(np.abs(f_off - np.asarray(ref_f)).mean())
                        if ref_f is not None else float("nan"))
                rows.append(dict(idx=i, formula=src.get_chemical_formula(), n=n,
                                 e_off=e_off, e_on=e_on, de_at=abs(e_off - e_on) / n,
                                 dfmax=float(np.abs(f_off - f_on).max()),
                                 t_off=t_off, t_on=t_on,
                                 spd=(t_off / t_on if t_on else float("nan")), f_dft_mae=fmae))
            except Exception as ex:  # noqa: BLE001 - record and continue
                rows.append(dict(idx=i, formula=src.get_chemical_formula(), n=n, error=str(ex)))
        all_rows[dtype] = rows

        ok = [r for r in rows if "error" not in r]
        bad = [r for r in rows if "error" in r]
        print(f"{'idx':>4}{'formula':>18}{'nat':>5}{'E_off[eV]':>16}{'E_on[eV]':>16}"
              f"{'dE/at[eV]':>11}{'dFmax':>9}{'t_off[ms]':>10}{'t_on[ms]':>10}{'speedup':>9}")
        for r in ok:
            print(f"{r['idx']:>4}{r['formula'][:18]:>18}{r['n']:>5}{r['e_off']:>16.6f}"
                  f"{r['e_on']:>16.6f}{r['de_at']:>11.2e}{r['dfmax']:>9.1e}"
                  f"{r['t_off'] * 1e3:>10.1f}{r['t_on'] * 1e3:>10.1f}{r['spd']:>8.2f}x")
        for r in bad:
            print(f"{r['idx']:>4}{r['formula'][:18]:>18}{r['n']:>5}  ERROR: {r['error'][:55]}")
        if ok:
            we = max(r["de_at"] for r in ok)
            wf = max(r["dfmax"] for r in ok)
            fmae_all = np.nanmean([r["f_dft_mae"] for r in ok])
            tot_off = sum(r["t_off"] for r in ok)
            tot_on = sum(r["t_on"] for r in ok)
            spds = [r["spd"] for r in ok]
            print("-" * 104)
            print(f"frames OK={len(ok)} ERR={len(bad)} | worst |dE/atom|={we:.2e} eV | "
                  f"worst |dF off-vs-on|={wf:.2e} eV/A | MACE-vs-DFT force MAE={fmae_all:.3f} eV/A")
            print(f"overall speedup = {tot_off / tot_on:.2f}x  "
                  f"(sum t_off {tot_off * 1e3:.0f} -> t_on {tot_on * 1e3:.0f} ms) | "
                  f"median/struct = {float(np.median(spds)):.2f}x | max = {float(np.max(spds)):.2f}x")
            print("speedup by size:")
            for lo, hi in [(0, 16), (16, 32), (32, 64), (64, 128), (128, 10**9)]:
                sub = [r for r in ok if lo <= r["n"] < hi]
                if sub:
                    to, tn = sum(r["t_off"] for r in sub), sum(r["t_on"] for r in sub)
                    label = f"{lo:>3}-{hi if hi < 10**9 else 'inf':<3} atoms"
                    print(f"  {label}: n={len(sub):>3}  speedup={to / tn:.2f}x")
            print("VERDICT:", "PASS — cueq lossless on MPtraj"
                  if (we < 1e-4 and wf < 1e-3 and not bad) else "CHECK — see deltas/errors above")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
