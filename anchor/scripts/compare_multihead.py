#!/usr/bin/env python3
"""Add the user's MULTIHEAD-replay fine-tune (job 85, mace_mh_mix_e0pbe86_multihead)
to the anchor-vs-FT force comparison — the SOTA forgetting-mitigation baseline the
publishability note said we hadn't compared against.

The multihead run only left a mid-training checkpoint (epoch-169 .pt, no exported
.model), so we first convert it via a same-architecture template (2-head MACE-MH),
then evaluate BOTH readout columns. Head NAMES differ between template and job-85
(template: Default/mp_pbe_refit_add ; job-85: pt_head/Default) and load_state_dict
matches by tensor key, so we identify the columns EMPIRICALLY by behaviour:
the base-preserving column (good MPtrj) = pt_head, the broken one = Default.

Evaluates forces (F R², F-MAE vs DFT) on:
  - keep_test (OOD short contacts, where the anchor's gate is active)
  - mixed_test all / distorted(<1.5 Å) / normal(≥1.5 Å)   (FT's own held-out split)
  - MPtrj-1k (pure base)
Writes results/multihead_compare.json (merged later into PAIRPHYS_REPORT §3.3/3.3a).
"""
from __future__ import annotations
import json
import os
import pathlib
import sys

import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

ROOT = pathlib.Path(os.environ.get("ZBL_IAML_WS", ""))  # TODO: set for your machine (external workspace)
HERE = ROOT / "idea_uncertainty_gated_physics_anchor"

# Re-exported via MACE --restart_latest at epoch 169 (no further training: loss == epoch-169);
# uncompiled (the compiled variant triggers a Global-alloc error per project memory).
OUT_MODEL = ROOT / ("mace_mh_mix_e0pbe86_multihead/results/mace_mh0_mix_e0pbe86_multihead/"
                    "checkpoints/mace_mh0_mix_e0pbe86_multihead_run-42.model")

KEEP = ROOT / "vasp_tier1/collected/preflight/splits/keep_test.xyz"
MIXED = ROOT / "mixed_dataset_clean/data/mixed_test.xyz"
MPTRJ = HERE / "results/mptrj1k.xyz"
OUTJ = HERE / "results/multihead_compare.json"


def get_ref(at):
    for k in ("REF_forces", "forces"):
        if k in at.arrays:
            return np.asarray(at.arrays[k], float)
    return None


def fmin(at):
    d = neighbor_list("d", at, 5.0)
    return float(d.min()) if len(d) else 9.0


def r2(pred, ref):
    p = np.concatenate([x.reshape(-1) for x in pred]); r = np.concatenate([x.reshape(-1) for x in ref])
    return float(1 - ((p - r) ** 2).sum() / ((r - r.mean()) ** 2).sum())


def mae(pred, ref):
    p = np.concatenate([x.reshape(-1) for x in pred]); r = np.concatenate([x.reshape(-1) for x in ref])
    return float(np.mean(np.abs(p - r)))


def convert():
    if not OUT_MODEL.exists():
        sys.exit(f"ERROR: re-exported multihead model not found: {OUT_MODEL}")
    print(f"using re-exported multihead model: {OUT_MODEL.name}")


def forces(calc, at):
    a = at.copy(); a.calc = calc
    return np.asarray(a.get_forces(), float)


def eval_set(calc, frames, splitter=None):
    """splitter(at)->group tag or None for single group 'all'."""
    pools = {}
    for at in frames:
        Fref = get_ref(at)
        if Fref is None:
            continue
        F = forces(calc, at)
        tags = ["all"] + ([splitter(at)] if splitter else [])
        for g in tags:
            pools.setdefault(g, {"ref": [], "pred": []})
            pools[g]["ref"].append(Fref); pools[g]["pred"].append(F)
    return {g: dict(n=len(p["ref"]), F_R2=r2(p["pred"], p["ref"]), F_MAE=mae(p["pred"], p["ref"]))
            for g, p in pools.items()}


def main():
    convert()
    template_model = torch.load(OUT_MODEL, map_location="cpu", weights_only=False)
    head_names = list(getattr(template_model, "heads", ["head0", "head1"]))
    print(f"converted model heads (template names): {head_names}")

    keep = read(str(KEEP), index=":")
    mixed = read(str(MIXED), index=":")
    mptrj = read(str(MPTRJ), index=":")
    print(f"keep={len(keep)} mixed={len(mixed)} mptrj={len(mptrj)}")

    out = {"heads_template_order": head_names, "by_head": {}}
    for h in head_names:
        print(f"\n=== head column '{h}' ===", flush=True)
        calc = MACECalculator(model_paths=[str(OUT_MODEL)], device="cuda",
                              default_dtype="float32", head=h)
        keep_all = eval_set(calc, keep)["all"]
        keep_ge = eval_set(calc, [a for a in keep if fmin(a) >= 0.30])["all"]
        mixed_res = eval_set(calc, mixed, lambda a: "distorted" if fmin(a) < 1.5 else "normal")
        mptrj_res = eval_set(calc, mptrj)["all"]
        out["by_head"][h] = {
            "keep_all": keep_all, "keep_ge0.30": keep_ge,
            "mixed": mixed_res, "mptrj": mptrj_res,
        }
        print(f"  keep all : R²{keep_all['F_R2']:+.3f} MAE{keep_all['F_MAE']:.3g}")
        print(f"  mptrj    : R²{mptrj_res['F_R2']:+.3f} MAE{mptrj_res['F_MAE']:.3g}")
        print(f"  mixed all: R²{mixed_res['all']['F_R2']:+.3f} MAE{mixed_res['all']['F_MAE']:.3g}")

    # empirically label: base-preserving head = highest MPtrj R² → 'pt_head'; other → 'Default'
    ranked = sorted(head_names, key=lambda h: out["by_head"][h]["mptrj"]["F_R2"], reverse=True)
    out["role"] = {ranked[0]: "pt_head (base-preserving)", ranked[1]: "Default (user-adapted)"}
    print(f"\nROLE: {ranked[0]} = pt_head(base) ; {ranked[1]} = Default(user)")
    OUTJ.write_text(json.dumps(out, indent=2))
    print("→", OUTJ)


if __name__ == "__main__":
    main()
