"""Focused, OOM-hardened probe: does compile+cueq actually RUN in mace 0.3.16?

Two-stage, both reported even if the second fails:
  (A) cheap: build the calculator (torch.compile is lazy) and inspect use_compile
      + any "compile disabled" warning  -> API-level engage signal, never OOMs.
  (B) heavy: attempt ONE compiled forward on a tiny 27-atom Cu cell, compare to
      the e3nn fp32 reference energy, time it.  Wrapped in try/except so an
      inductor OOM/crash still leaves stage-A reported.

Run ONE combo per process/srun (arg --combo) so a crash is isolated and each
gets the full memory allocation.  Inductor compile-thread count is capped via
TORCHINDUCTOR_COMPILE_THREADS to bound host RAM.
"""
import os, sys, time, json, warnings, logging, argparse, traceback

warnings.filterwarnings("ignore")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # cap compile RAM

import numpy as np
import torch
from ase.build import bulk

MODEL = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
HEAD = "omat_pbe"
COMBOS = {  # name -> (enable_cueq, compile_mode)
    "e3nn": (False, None),
    "cueq": (True, None),
    "cmp-e3nn": (False, "default"),
    "cmp+cueq": (True, "default"),
}

_warn_hits = []
class _H(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "compile" in m.lower():
            _warn_hits.append(m)
logging.getLogger().addHandler(_H())
logging.getLogger().setLevel(logging.WARNING)


def cu_cell(nn, seed=0):
    a = bulk("Cu", "fcc", a=3.61).repeat((nn, nn, nn))
    a.positions += np.random.default_rng(seed).normal(scale=0.05, size=a.positions.shape)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combo", required=True, choices=list(COMBOS))
    p.add_argument("--dtype", default="float32")
    p.add_argument("--nn", type=int, default=3)  # 27-atom cell by default
    args = p.parse_args()

    import mace
    from mace.calculators import MACECalculator
    en, cm = COMBOS[args.combo]
    res = {"combo": args.combo, "dtype": args.dtype, "nn": args.nn,
           "mace": mace.__version__, "torch": torch.__version__,
           "enable_cueq": en, "compile_mode": cm}

    # ---- stage A: cheap build (lazy compile) ----
    _warn_hits.clear()
    try:
        calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype=args.dtype,
                              head=HEAD, enable_cueq=en, compile_mode=cm, warmup=False)
        res["build_ok"] = True
        res["use_compile"] = getattr(calc, "use_compile", "n/a")
        res["warn_compile_disabled"] = any(
            ("not support" in w.lower() or "setting compile_mode" in w.lower()
             or "disabl" in w.lower()) for w in _warn_hits)
        res["compile_warns"] = _warn_hits[:5]
    except Exception as ex:
        res["build_ok"] = False
        res["build_error"] = str(ex).strip().splitlines()[-1][:200]
        print("RESULT_JSON: " + json.dumps(res)); return

    # ---- stage B: one real (possibly compiled) forward ----
    atoms = cu_cell(args.nn); atoms.calc = calc
    try:
        t0 = time.perf_counter()
        e = float(atoms.get_potential_energy())  # triggers compile on first call
        torch.cuda.synchronize()
        res["forward_ok"] = True
        res["energy_first_call_s"] = time.perf_counter() - t0
        res["energy"] = e
        res["natoms"] = len(atoms)
        # a second call to time the steady-state (post-compile) cost
        t1 = time.perf_counter(); float(atoms.get_potential_energy())
        torch.cuda.synchronize()
        res["energy_second_call_s"] = time.perf_counter() - t1
    except Exception as ex:
        res["forward_ok"] = False
        res["forward_error"] = str(ex).strip().splitlines()[-1][:200]
        res["forward_traceback"] = traceback.format_exc()[-800:]

    print("RESULT_JSON: " + json.dumps(res))


if __name__ == "__main__":
    main()
