"""Single-combo worker for the mace 0.3.16 validation.

Runs exactly ONE MACECalculator per process so host RAM + GPU are fully
reclaimed by process exit between combos. This isolates the torch.compile /
inductor memory blowup that OOM-killed the monolithic script, and gives the
cleanest signal for "does compile+cueq engage" (compile state is per-process).

Emits one machine-readable line:  RESULT_JSON: {...}
"""
import os, sys, time, json, warnings, logging, argparse, traceback

warnings.filterwarnings("ignore")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
from ase.io import read
from ase.build import bulk

MODEL = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
HEAD = "omat_pbe"
BASE_JSON = "/path/to/cueq_float64_off.json"  # TODO: set for your machine
STRUCT = "/path/to/structures_large.extxyz"  # TODO: set for your machine

# combo name -> (enable_cueq, compile_mode)
COMBOS = {
    "e3nn": (False, None),
    "cueq": (True, None),
    "cmp-e3nn": (False, "default"),
    "cmp+cueq": (True, "default"),
}

# capture any "compile"-related warning routed through logging
_warn_hits = []
class _H(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "compile" in m.lower():
            _warn_hits.append(m)
logging.getLogger().addHandler(_H())
logging.getLogger().setLevel(logging.WARNING)


def build(enable_cueq, compile_mode, dtype):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=MODEL, device="cuda", default_dtype=dtype,
                          head=HEAD, enable_cueq=enable_cueq, compile_mode=compile_mode)


def energy(calc, atoms):
    a = atoms.copy(); a.calc = calc
    return float(a.get_potential_energy())


def cu_cell(nn, seed=0):
    a = bulk("Cu", "fcc", a=3.61).repeat((nn, nn, nn))
    a.positions += np.random.default_rng(seed).normal(scale=0.05, size=a.positions.shape)
    return a


def task_lossless():
    """0.3.16 e3nn fp64 energies for every labelled frame present in the ref."""
    ref = {r["label"]: r for r in json.load(open(BASE_JSON))["results"]}
    frames = read(STRUCT, index=":")
    calc = build(False, None, "float64")
    out = []
    for a in frames:
        lab = a.info.get("label", a.get_chemical_formula())
        if lab not in ref:
            continue
        out.append({"label": lab, "natoms": len(a), "e16": energy(calc, a),
                    "e15": ref[lab]["energy"]})
    return {"task": "lossless", "results": out}


def task_engage(combo, dtype):
    """Build one combo, run a 125-atom Cu cell, report whether compile engaged."""
    en, cm = COMBOS[combo]
    _warn_hits.clear()
    atoms = cu_cell(5)
    res = {"task": "engage", "combo": combo, "dtype": dtype,
           "enable_cueq": en, "compile_mode": cm}
    try:
        calc = build(en, cm, dtype)
        res["use_compile"] = getattr(calc, "use_compile", "n/a")
        res["energy"] = energy(calc, atoms)
        res["natoms"] = len(atoms)
        res["warn_compile_disabled"] = any(
            ("not support" in w.lower() or "setting compile_mode" in w.lower()
             or "disabl" in w.lower())
            for w in _warn_hits)
        res["compile_warns"] = _warn_hits[:5]
        res["ok"] = True
    except Exception as ex:
        res["ok"] = False
        res["error"] = str(ex).strip().splitlines()[-1][:160]
        res["traceback"] = traceback.format_exc()[-600:]
    return res


def task_bench(combo, dtype, sizes):
    """Time one combo across several Cu supercell sizes (build once, reuse)."""
    en, cm = COMBOS[combo]
    res = {"task": "bench", "combo": combo, "dtype": dtype, "times_ms": {}}
    try:
        calc = build(en, cm, dtype)
    except Exception as ex:
        res["ok"] = False
        res["error"] = str(ex).strip().splitlines()[-1][:160]
        return res

    def timed(atoms, n=12, w=6):
        a = atoms.copy(); a.calc = calc
        for _ in range(w):
            calc.calculate(a)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            calc.calculate(a)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    for nn in sizes:
        a = cu_cell(nn)
        try:
            res["times_ms"][str(len(a))] = timed(a) * 1e3
        except Exception as ex:
            res["times_ms"][str(len(a))] = None
            res.setdefault("errors", {})[str(len(a))] = str(ex)[:120]
    res["ok"] = True
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["lossless", "engage", "bench"])
    p.add_argument("--combo", default="e3nn", choices=list(COMBOS))
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--sizes", default="3,5,7,9")
    args = p.parse_args()

    import mace
    sys.stderr.write(f"[worker] mace {mace.__version__} torch {torch.__version__} "
                     f"task={args.task} combo={args.combo} dtype={args.dtype}\n")

    if args.task == "lossless":
        res = task_lossless()
    elif args.task == "engage":
        res = task_engage(args.combo, args.dtype)
    else:
        sizes = tuple(int(x) for x in args.sizes.split(","))
        res = task_bench(args.combo, args.dtype, sizes)

    print("RESULT_JSON: " + json.dumps(res))


if __name__ == "__main__":
    main()
