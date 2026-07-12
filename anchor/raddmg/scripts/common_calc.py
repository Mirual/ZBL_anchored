#!/usr/bin/env python3
"""Tier-0 calculator factory: the two models under test.

  vanilla          — MACE-MH-0 as-is (ZBL on)
  vanilla_pairphys — MACE-MH-0 + RND-gated per-pair ZBL-residual anchor (core_zbl on)

Thin wrapper over md_stability/scripts/anchor_calculator.AnchorCalculator so every
Tier-0 driver is calculator-agnostic.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
sys.path.insert(0, str(ROOT / "md_stability" / "scripts"))
from anchor_calculator import AnchorCalculator  # noqa: E402

CACHE = str(ROOT / "results" / "dimer_tables" / "dimer_zblON_user_wbm.pkl")
TAGS = ("vanilla", "vanilla_pairphys")


def make_calculator(tag: str, device: str = "cuda"):
    if tag == "vanilla":
        return AnchorCalculator(mode="vanilla", device=device, disable_zbl=False)
    if tag == "vanilla_pairphys":
        # core_zbl=False: MACE-MH-0 ALREADY carries a built-in ZBL, so the foundation
        # owns the deep core (r < dimer-cache floor 0.30 Å). Turning core_zbl ON here
        # adds a SECOND full analytic ZBL on top of the foundation's → double-counting,
        # which injects 10^4 eV/Å force spikes below 0.30 Å and *regresses* forces
        # (keep_test R² 0.650→0.407). With it OFF the sub-floor residual stays bounded
        # and the anchor improves forces (keep_test R² 0.650→0.829). core_zbl should
        # only be ON when the foundation's own ZBL is disabled (disable_zbl=True).
        return AnchorCalculator(mode="pairphys", device=device, disable_zbl=False,
                                core_zbl=False, dimer_cache_path=CACHE)
    raise ValueError(f"unknown calculator tag {tag!r}; use one of {TAGS}")
