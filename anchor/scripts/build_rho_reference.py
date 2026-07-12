#!/usr/bin/env python3
"""Reference for ρ (extrapolation-score): MPtrj-train descriptor bank + PCA + kNN-calibration.

ρ_i = "how FAR atom i's local environment is from the foundation's training distribution".
Computed via kNN-distance in the PCA-compressed space of MACE per-atom descriptors (256-dim).
Reference = MPtrj (what the foundation was trained on) → in-distribution.

Output: results/rho_reference.npz {pca_mean, pca_comp, ref (proj, [M,K]), r_lo, r_hi}.
Calibration: kNN-distance of the reference atoms themselves → r_lo=p90, r_hi=p99.5 (smoothstep threshold).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
from sklearn.neighbors import NearestNeighbors

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
MIXED = str(Path(os.environ.get("ZBL_MIXED_DATA", "/path/to/mixed_dataset/data")) / "mixed_train.xyz")
OUT = Path(__file__).resolve().parents[1] / "results" / "rho_reference.npz"
N_FRAMES = 1500         # MPtrj frames for the reference (more → fewer false-positive OOD)
N_ATOMS_REF = 40000     # subsample of reference atoms
PCA_K = 32
KNN = 8


def descriptors(frames, calc):
    D = []
    for at in frames:
        D.append(np.asarray(calc.get_descriptors(at, invariants_only=True)))
    return np.concatenate(D, axis=0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32",
                          head="mp_pbe_refit_add")
    frames = [a for a in read(MIXED, index=":") if a.info["REF_energy"] / len(a) < 50][:N_FRAMES]
    print(f"reference: {len(frames)} MPtrj frames")
    X = descriptors(frames, calc)              # [Ntot, 256]
    print(f"atom descriptors: {X.shape}")

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), min(N_ATOMS_REF, len(X)), replace=False)
    Xr = X[idx]
    mean = Xr.mean(0)
    U, S, Vt = np.linalg.svd(Xr - mean, full_matrices=False)
    comp = Vt[:PCA_K]                           # [K, 256]
    ref = (Xr - mean) @ comp.T                  # [M, K]

    nn = NearestNeighbors(n_neighbors=KNN + 1).fit(ref)
    dist, _ = nn.kneighbors(ref)                # self → drop col 0
    rho_self = dist[:, 1:].mean(1)
    r_lo, r_hi = np.percentile(rho_self, [90, 99.5])
    print(f"kNN self-dist: med={np.median(rho_self):.3f}  r_lo(p90)={r_lo:.3f}  r_hi(p99.5)={r_hi:.3f}")

    np.savez(OUT, pca_mean=mean, pca_comp=comp, ref=ref,
             r_lo=float(r_lo), r_hi=float(r_hi), knn=KNN)
    print(f"saved {OUT}  (ref {ref.shape}, K={PCA_K})")


if __name__ == "__main__":
    main()
