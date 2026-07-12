"""Aligned head-to-head: ALL 5 DPA-3.1 variants on the SAME frames (keep_test=OOD compressed + MPtrj=base),
forces F R² + F MAE vs DFT. Variants: vanilla / pure ZBL (bolt-on, no FT) / FT mix / FT user-only / anchor
(vanilla + RND-gated per-pair corr). Env: deepmd_env. Run via slurm/35."""
import json
import numpy as np
from ase.io import read
from deepmd.calculator import DP
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpa_common import FOUNDATION, compute_descriptors, load_dp
from gate import RNDGate
from predict import corr
from pair_physics import DimerCache

KEEP = os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight") + "/keep_test.xyz"
MPTRJ = os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz")
CKPTS = {
    "vanilla (no ZBL)": FOUNDATION,
    # TODO: set for your machine
    "pure ZBL (bolt-on)": os.environ.get("ZBL_DPA_ZBL_INIT", "/path/to/dpa-3.1-3m-ft-zbl-init.pth"),
    # TODO: set for your machine
    "FT mix_e0pbe86": os.environ.get("ZBL_DPA_FT_MIX", "/path/to/dpa31_ft_mix_e0pbe86.pth"),
    # TODO: set for your machine
    "FT user-only (+ZBL+F100E1)": os.environ.get("ZBL_DPA_FT_USER", "/path/to/dpa31_ft_user_only.pth"),
}
_SM = bool(os.environ.get("HTH_SMOKE"))                  # quick smoke: 3 frames/split
SPLITS = {"distorted (compressed OOD)": read(KEEP, index=":3" if _SM else ":"),
          "MPtrj (baseline)": read(MPTRJ, index=":3" if _SM else ":500")}
print({k: len(v) for k, v in SPLITS.items()})


def fref(at):
    for k in ("REF_forces", "forces"):
        if k in at.arrays:
            return np.asarray(at.arrays[k], float)
    return None


def predict_dp(ckpt, frames):
    """list of [N,3] forces (None if OOV/failure)."""
    import os
    if not os.path.exists(ckpt):
        print(f"  ckpt MISSING: {ckpt}"); return None
    dp = DP(model=ckpt).dp
    td = {s: i for i, s in enumerate(dp.get_type_map())}
    out = []
    for at in frames:
        syms = at.get_chemical_symbols()
        if any(s not in td for s in syms):
            out.append(None); continue
        try:
            coords = at.get_positions().reshape(1, -1)
            cells = np.asarray(at.get_cell()).reshape(1, -1) if sum(at.get_pbc()) > 0 else None
            atype = np.array([td[s] for s in syms], dtype=np.int32)
            _, f, _ = dp.eval(coords, cells, atype)[:3]
            out.append(np.asarray(f[0]))
        except Exception:                                # noqa: BLE001
            out.append(None)
    del dp
    return out


def fmetrics(frames, pf):
    if pf is None:
        return None, None
    fr, fp = [], []
    for at, p in zip(frames, pf):
        r = fref(at)
        if p is None or r is None:
            continue
        fr.append(r.reshape(-1)); fp.append(np.asarray(p).reshape(-1))
    if not fr:
        return None, None
    fr = np.concatenate(fr); fp = np.concatenate(fp)
    mae = float(np.abs(fp - fr).mean())
    r2 = float(1 - ((fp - fr) ** 2).sum() / ((fr - fr.mean()) ** 2).sum())
    return mae, r2


# anchor infra (one-time)
_ANCHOR_RES = os.environ.get("ZBL_ANCHOR_RESULTS", "results")
gate = RNDGate("cuda"); dc = DimerCache(load_dp(FOUNDATION), cache_path=_ANCHOR_RES + "/dimer_dpa.pkl")
th = json.load(open(_ANCHOR_RES + "/pairphys_theta.json"))
rlo, rhi, lam, pw = th["r_lo"], th["r_hi"], th["lam"], th["power"]

results = {}
for split, frames in SPLITS.items():
    print(f"\n=== split: {split} ({len(frames)} frames) ===")
    van_F = predict_dp(FOUNDATION, frames)               # reuse for vanilla + anchor base
    for name, ckpt in CKPTS.items():
        pf = van_F if name.startswith("vanilla") else predict_dp(ckpt, frames)
        mae, r2 = fmetrics(frames, pf)
        results.setdefault(name, {})[split] = dict(F_MAE=mae, F_R2=r2)
        print(f"  {name:32s}: F R²={r2}  F MAE={mae}")
    # anchor = vanilla + corr
    anc = []
    for at, vf in zip(frames, van_F):
        if vf is None:
            anc.append(None); continue
        nov = gate.novelty(compute_descriptors([at])[0])
        _, dFc = corr(at, nov, dc, rlo, rhi, lam, pw)
        anc.append(vf + dFc)
    mae, r2 = fmetrics(frames, anc)
    results.setdefault("anchor (RND-gated, no FT)", {})[split] = dict(F_MAE=mae, F_R2=r2)
    print(f"  {'anchor (RND-gated, no FT)':32s}: F R²={r2}  F MAE={mae}")

json.dump(results, open(_ANCHOR_RES + "/head_to_head_aligned.json", "w"), indent=2)
print("\n===== ALIGNED HEAD-TO-HEAD (same frames) =====")
order = ["vanilla (no ZBL)", "pure ZBL (bolt-on)", "FT mix_e0pbe86", "FT user-only (+ZBL+F100E1)", "anchor (RND-gated, no FT)"]
print(f"{'variant':32s} | {'keep F R²':>10s} | {'keep MAE':>9s} | {'MPtrj F R²':>11s} | {'MPtrj MAE':>9s}")
for name in order:
    r = results.get(name, {})
    k = r.get("distorted (compressed OOD)", {}); m = r.get("MPtrj (baseline)", {})
    def fmt(x, p=3): return f"{x:.{p}f}" if isinstance(x, float) else "—"
    print(f"{name:32s} | {fmt(k.get('F_R2')):>10s} | {fmt(k.get('F_MAE'),1):>9s} | "
          f"{fmt(m.get('F_R2')):>11s} | {fmt(m.get('F_MAE'),3):>9s}")
print("saved results/head_to_head_aligned.json")
