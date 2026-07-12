"""#6 payoff benchmark: does torch.compile add steady-state throughput ON TOP
of cueq in mace 0.3.16 (now that compile+cueq engages)?

One combo + one fixed size per process (compile recompiles per shape, and is
RAM-heavy -> isolate). Pays the one-time compile during a long warmup, then
times steady-state single-point energy+forces.
"""
import os, sys, time, json, warnings, argparse, traceback
warnings.filterwarnings("ignore")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import torch
from ase.build import bulk

MODEL = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
HEAD = "omat_pbe"
COMBOS = {"e3nn": (False, None), "cueq": (True, None),
          "cmp-e3nn": (False, "default"), "cmp+cueq": (True, "default")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combo", required=True, choices=list(COMBOS))
    p.add_argument("--nn", type=int, default=8)        # 1024-atom Cu
    p.add_argument("--dtype", default="float32")
    p.add_argument("--warm", type=int, default=15)
    p.add_argument("--n", type=int, default=40)
    args = p.parse_args()

    import mace
    from mace.calculators import MACECalculator
    en, cm = COMBOS[args.combo]

    a = bulk("Cu", "fcc", a=3.61).repeat((args.nn,) * 3)
    a.positions += np.random.default_rng(0).normal(scale=0.05, size=a.positions.shape)
    res = {"combo": args.combo, "natoms": len(a), "dtype": args.dtype,
           "mace": mace.__version__}
    try:
        calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype=args.dtype,
                              head=HEAD, enable_cueq=en, compile_mode=cm, warmup=False)
        res["use_compile"] = getattr(calc, "use_compile", "n/a")
        a.calc = calc
        t0 = time.perf_counter()
        for _ in range(args.warm):                      # pay one-time compile here
            calc.calculate(a)
        torch.cuda.synchronize()
        res["warm_total_s"] = time.perf_counter() - t0
        res["energy"] = float(a.get_potential_energy())

        torch.cuda.synchronize(); t1 = time.perf_counter()
        for _ in range(args.n):
            calc.calculate(a)
        torch.cuda.synchronize()
        res["steady_ms"] = (time.perf_counter() - t1) / args.n * 1e3
        res["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 1e6
        res["ok"] = True
    except Exception as ex:
        res["ok"] = False
        res["error"] = str(ex).strip().splitlines()[-1][:200]
        res["traceback"] = traceback.format_exc()[-600:]
    print("RESULT_JSON: " + json.dumps(res))


if __name__ == "__main__":
    main()
