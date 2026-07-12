#!/usr/bin/env python3
"""Extended high-pressure: vanilla(ZBL) vs pairphys(ZBL, deploy) on a set of distorted structures.
Loads both calculators ONCE, runs a compression ramp over all of them, computes statistics
'on what % anchor prevents the collapse that vanilla does not survive'."""
from __future__ import annotations
import argparse, json, glob, os, sys
from pathlib import Path
import numpy as np
from ase.io import read
sys.path.insert(0, str(Path(__file__).resolve().parent))
if os.environ.get("ZBL_IAML_WS"):
    sys.path.insert(0, os.path.join(os.environ["ZBL_IAML_WS"], "idea_uncertainty_gated_physics_anchor/md_stability/scripts"))
from anchor_calculator import AnchorCalculator   # noqa: E402
from mlip_arena_highP import ramp                 # noqa: E402

CACHE = os.path.join(os.environ.get("ZBL_ANCHOR_RESULTS", "results"), "dimer_tables/dimer_zblON_user_wbm.pkl")


def crash_scale(rampres, floor):
    last = rampres[-1]
    return last["scale"] if last["crash"] else floor


def main():
    p = argparse.ArgumentParser()
    # TODO: set for your machine
    p.add_argument("--systems", default=os.environ.get("ZBL_HIGHP_SYSTEMS", "/path/to/highP_systems"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cache = CACHE if Path(CACHE).exists() else None
    van = AnchorCalculator(mode="vanilla", device="cuda", disable_zbl=False)
    pp = AnchorCalculator(mode="pairphys", device="cuda", disable_zbl=False, dimer_cache_path=cache)
    scales = np.linspace(1.0, 0.55, 19); floor = float(scales[-1])
    files = sorted(glob.glob(f"{args.systems}/*.xyz"))
    rows = []
    n_van_crash = n_pp_crash = n_saved = 0
    for f in files:
        at = read(f)
        rv = ramp(van, at, scales); rp = ramp(pp, at, scales)
        csv = crash_scale(rv, floor); csp = crash_scale(rp, floor)
        vc = rv[-1]["crash"]; pc = rp[-1]["crash"]
        n_van_crash += vc; n_pp_crash += pc
        saved = vc and not pc
        n_saved += saved
        rows.append(dict(system=Path(f).stem, n_atoms=len(at),
                         vanilla_crash=bool(vc), vanilla_crash_scale=float(csv),
                         pairphys_crash=bool(pc), pairphys_crash_scale=float(csp), anchor_saved=bool(saved)))
        print(f"{Path(f).stem}: van {'CRASH@%.3f'%csv if vc else 'ok'} | pp {'CRASH@%.3f'%csp if pc else 'ok'}"
              f"{'  ← ANCHOR SAVED' if saved else ''}", flush=True)
    n = len(files)
    summ = dict(n=n, vanilla_crashes=n_van_crash, pairphys_crashes=n_pp_crash, anchor_saved=n_saved,
                pct_vanilla_crash=100*n_van_crash/n, pct_pairphys_crash=100*n_pp_crash/n,
                pct_anchor_saved=100*n_saved/n)
    Path(args.out).write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    print(f"\nTOTAL n={n}: vanilla crash {n_van_crash} ({summ['pct_vanilla_crash']:.0f}%), "
          f"pairphys crash {n_pp_crash} ({summ['pct_pairphys_crash']:.0f}%), "
          f"anchor saved {n_saved} ({summ['pct_anchor_saved']:.0f}%) → {args.out}", flush=True)


if __name__ == "__main__":
    main()
