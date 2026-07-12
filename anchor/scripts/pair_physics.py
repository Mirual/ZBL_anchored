#!/usr/bin/env python3
"""Layer 1: per-pair physical residual on top of a pretrained MACE-MH-0 (NO retraining).

Correction = the missing ZBL repulsion, RELATIVE to the model's own dimer:
    ΔV_ij(r) = [ V_ZBL(Z_i,Z_j,r) − V_MACE-dimer(Z_i,Z_j,r) ]_+ · f_cut(r; r_cut)
- [·]_+ : repulsion only (cannot add attraction → cannot break a valid structure).
- subtract the model's OWN dimer: MACE-MH-0 already has a built-in ZBL → we add only what is MISSING.
- f_cut → 0 above r_cut = κ·(r_cov_i+r_cov_j) → self-vanishing at the equilibrium bond (valid bonds untouched).

DimerCache computes the model's curve on a grid lazily (only for encountered pairs) and caches it.
"""
from __future__ import annotations
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

K_E = 14.399645          # eV·Å  (e²/4πε₀)
# universal ZBL coefficients
_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
_D = np.array([3.19980, 0.94229, 0.40290, 0.20162])


def zbl_V(Zi: int, Zj: int, r: np.ndarray) -> np.ndarray:
    """ZBL screened-Coulomb repulsion, eV."""
    a = 0.46850 / (Zi ** 0.23 + Zj ** 0.23)
    x = r[:, None] / a
    phi = (_C * np.exp(-_D * x)).sum(1)
    return K_E * Zi * Zj / r * phi


def zbl_grad(Zi: int, Zj: int, r: np.ndarray):
    """Analytic ZBL: V(r), eV and dV/dr, eV/Å (diverge as 1/r as r→0; exact at any r)."""
    a = 0.46850 / (Zi ** 0.23 + Zj ** 0.23)
    x = r[:, None] / a
    e = np.exp(-_D * x)
    phi = (_C * e).sum(1)
    phip = (_C * (-_D) * e).sum(1)           # dphi/dx
    V = K_E * Zi * Zj / r * phi
    dVdr = K_E * Zi * Zj * (-phi / r ** 2 + phip / (a * r))
    return V, dVdr


def _smoothstep(r, lo, hi):                       # 1 for r<lo, 0 for r>hi
    t = np.clip((r - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - (t * t * (3 - 2 * t))


class DimerCache:
    """Lazy cache of per-pair residual ΔV(r) and dΔV/dr (via the model's own dimer)."""

    def __init__(self, calc, kappa=0.90, width=0.40, grid_step=0.05, r_ref=6.0, cache_path=None):
        self.calc = calc
        self.kappa = kappa          # r_cut = kappa·(rcov_i+rcov_j)
        self.width = width          # width of f_cut
        self.gstep = grid_step
        self.r_ref = r_ref          # "separated" dimer as zero interaction
        # cache_path: on-disk table of dimer curves (depends ONLY on model+pair, not on structures).
        # precompute once → reuse across all processes (calibration/inference) without recomputation.
        import pickle
        from pathlib import Path as _P
        self.cache_path = _P(cache_path) if cache_path else None
        self._dirty = False
        if self.cache_path and self.cache_path.exists():
            self._cache = pickle.loads(self.cache_path.read_bytes())
        else:
            self._cache = {}

    def _mace_dimer(self, Zi, Zj, rgrid):
        """V_MACE(r) = E_dimer(r) − E_dimer(r_ref), eV (interaction relative to separated atoms)."""
        E = np.empty(len(rgrid))
        for k, r in enumerate(rgrid):
            at = Atoms(numbers=[Zi, Zj], positions=[[0, 0, 0], [r, 0, 0]],
                       cell=[20, 20, 20], pbc=False)
            at.calc = self.calc
            E[k] = at.get_potential_energy()
        at = Atoms(numbers=[Zi, Zj], positions=[[0, 0, 0], [self.r_ref, 0, 0]],
                   cell=[20, 20, 20], pbc=False)
        at.calc = self.calc
        return E - at.get_potential_energy()

    def get(self, Zi, Zj):
        key = (min(Zi, Zj), max(Zi, Zj))
        if key in self._cache:
            return self._cache[key]
        zi, zj = key
        r_cut = self.kappa * (covalent_radii[zi] + covalent_radii[zj])
        rgrid = np.arange(0.30, r_cut + self.width + self.gstep, self.gstep)
        v_zbl = zbl_V(zi, zj, rgrid)
        v_mace = self._mace_dimer(zi, zj, rgrid)
        residual = np.clip(v_zbl - v_mace, 0.0, None)        # only the missing repulsion
        fcut = _smoothstep(rgrid, r_cut - self.width, r_cut)  # self-vanishing at equilibrium
        dV = residual * fcut
        dVdr = np.gradient(dV, rgrid)
        entry = dict(r=rgrid, dV=dV, dVdr=dVdr, r_cut=float(r_cut))
        self._cache[key] = entry
        self._dirty = True
        return entry

    def save(self, path=None):
        """Save the dimer-curves table to disk (pickle). Called after precompute."""
        import pickle
        from pathlib import Path as _P
        p = _P(path) if path else self.cache_path
        if p is None:
            raise ValueError("no cache_path given")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(self._cache))
        self._dirty = False
        return p


def pair_correction(atoms, cache: DimerCache):
    """Σ ΔV over close pairs + analytic forces. Returns (E_corr, F_corr[N,3])."""
    n = len(atoms)
    F = np.zeros((n, 3))
    # global cutoff = max r_cut among present pairs
    Zs = np.unique(atoms.numbers)
    rmax = max(cache.kappa * (covalent_radii[a] + covalent_radii[b]) + cache.width
               for a in Zs for b in Zs)
    i, j, d, D = neighbor_list("ijdD", atoms, float(rmax))
    if len(d) == 0:
        return 0.0, F
    Znum = atoms.numbers
    Etot = 0.0
    for pair_key in {(min(Znum[a], Znum[b]), max(Znum[a], Znum[b])) for a, b in zip(i, j)}:
        e = cache.get(*pair_key)
        zi, zj = pair_key
        m = ((Znum[i] == zi) & (Znum[j] == zj)) | ((Znum[i] == zj) & (Znum[j] == zi))
        if not m.any():
            continue
        dm = d[m]
        dV = np.interp(dm, e["r"], e["dV"], left=e["dV"][0], right=0.0)
        dVdr = np.interp(dm, e["r"], e["dVdr"], left=e["dVdr"][0], right=0.0)
        Etot += 0.5 * dV.sum()
        np.add.at(F, i[m], (dVdr / dm)[:, None] * D[m])
    return float(Etot), F
