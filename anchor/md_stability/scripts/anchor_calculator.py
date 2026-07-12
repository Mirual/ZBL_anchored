#!/usr/bin/env python3
"""ASE Calculator: MACE-MH-0 + RND-gate + correction (vanilla | bornmayer | pairphys).

Wraps the foundation model so that ASE MD gets E,F = vanilla + gated-anchor at each step.
mode:
  vanilla   — MACE-MH-0 as is
  bornmayer — RND-gate(r_lo) × global Born–Mayer(A,b)
  pairphys  — RND-gate(r_lo) × per-pair ZBL-residual(DimerCache) × λ
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator
from e3nn import o3
from mace.modules.utils import extract_invariant

PARENT = Path(__file__).resolve().parents[2] / "scripts"   # idea/scripts
sys.path.insert(0, str(PARENT))
from rnd_anchor_predict import RNDGate                       # noqa: E402
from rho_anchor_predict import pair_corr_gated               # noqa: E402
from rnd_pairphys_predict import corr as pairphys_corr       # noqa: E402
from anchor_predict import smoothstep                        # noqa: E402
from pair_physics import DimerCache                          # noqa: E402

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")


class AnchorCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, mode="vanilla", device="cuda",
                 r_lo=0.05, r_hi=0.5, A=1200.0, b=0.5, ra=0.3, rb=1.5,
                 power=2.0, lam=0.4, disable_zbl=False, core_zbl=False,
                 dimer_cache_path=None, **kw):
        super().__init__(**kw)
        assert mode in ("vanilla", "bornmayer", "pairphys")
        self.mode = mode
        self.core_zbl = core_zbl
        self.par = dict(r_lo=r_lo, r_hi=r_hi, A=A, b=b, ra=ra, rb=rb, power=power, lam=lam)
        self.mace = MACECalculator(model_paths=[VAN], device=device,
                                   default_dtype="float32", head="mp_pbe_refit_add")
        if disable_zbl:   # del → forward checks hasattr(self,"pair_repulsion") → ZBL is skipped
            for mdl in self.mace.models:
                if hasattr(mdl, "pair_repulsion"):
                    del mdl.pair_repulsion
        self.gate = None if mode == "vanilla" else RNDGate(device)
        # dimer_cache_path: precomputed table of dimer curves (one per model, loaded from disk)
        self.dc = DimerCache(self.mace, cache_path=dimer_cache_path) if mode == "pairphys" else None
        # fused descriptors: capture node_feats with a hook during the E/F pass, to avoid running MACE
        # a 2nd time for get_descriptors (this gives ~2x anchor-MD speedup; E/F is untouched).
        self._nf = None
        if mode != "vanilla":
            m0 = self.mace.models[0]
            self._ni = int(m0.num_interactions)
            _irr = o3.Irreps(str(m0.products[0].linear.irreps_out))
            self._lmax = _irr.lmax
            self._ninv = _irr.dim // (self._lmax + 1) ** 2
            self._dim = _irr.dim

            def _grab(_mod, _in, out):
                if isinstance(out, dict) and "node_feats" in out:
                    self._nf = out["node_feats"]

            m0.register_forward_hook(_grab)

    def _descriptors(self, probe):
        """Invariant node descriptors for the gate from node_feats captured during the E/F pass
        (without a 2nd MACE pass). Postprocessing is identical to MACECalculator.get_descriptors; if
        the capture is missing — safe fallback to get_descriptors."""
        # safety: the DimerCache runs small 2-atom forwards through the same MACE model, which would
        # overwrite the hooked node_feats. In calculate() _descriptors is called BEFORE any dimer
        # forward, so the capture is the full system — but guard on atom count and fall back if not.
        if self._nf is None or int(self._nf.shape[0]) != len(probe):
            return self.mace.get_descriptors(probe)
        d = extract_invariant(self._nf, num_layers=self._ni,
                              num_features=self._ninv, l_max=self._lmax)
        per_layer = [self._dim] * self._ni
        per_layer[-1] = self._ninv
        to_keep = int(np.sum(per_layer[:self._ni]))
        return d[:, :to_keep].detach().cpu().numpy()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        at = self.atoms
        probe = at.copy(); probe.calc = self.mace
        e = float(probe.get_potential_energy()); f = np.asarray(probe.get_forces())
        if at.cell.rank == 3 and any(at.pbc):
            # MACE-stress (anchor contribution to the virial ≈0 in-distribution, where the gate is silent)
            try:
                self.results["stress"] = np.asarray(probe.get_stress())
            except Exception:
                pass
        if self.mode != "vanilla":
            nov = self.gate.novelty(self._descriptors(probe))
            p = self.par
            if self.mode == "bornmayer":
                rho = smoothstep(nov, p["r_lo"], p["r_hi"])
                ec, fc = pair_corr_gated(at, rho, p["A"], p["b"], p["ra"], p["rb"], p["power"])
            else:
                ec, fc = pairphys_corr(at, nov, self.dc, p["r_lo"], p["r_hi"], p["lam"], p["power"],
                                       core_zbl=self.core_zbl)
            e += ec; f = f + fc
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = f
