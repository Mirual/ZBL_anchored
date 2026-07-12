# ZBL_anchored — an uncertainty-gated physical anchor for foundation MLIPs

Foundation machine-learned interatomic potentials (MLIPs) such as **MACE-MH** and
**DPA-3.1** are trained on near-equilibrium structures. They *soften* in
high-energy / distorted regions — compressed bonds, defects, radiation cascades —
where there is little or no training data, and produce unphysically low forces
there. Built-in ZBL repulsion only covers the very-short-range limit (r → 0); the
compressed-but-bonded zone (~0.5–2 Å) is left to the GNN to extrapolate.

This repository implements a lightweight fix that needs **no retraining and does
not touch the base model's weights**:

$$
E \;=\; E_\text{GNN} \;+\; \sum_{i<j} w\!\big(\text{novelty}_{ij}\big)\; V_\text{phys}(r_{ij})
$$

- **`novelty`** — a per-atom extrapolation score computed from the foundation
  model's *own* latent features (a Random Network Distillation gate, ported from
  RL exploration). It is large exactly where the model is out-of-distribution.
- **`w(novelty)`** — a smooth gate calibrated with a SelectiveNet-style
  risk–coverage criterion: `w = 0` where the model is confident (predictions stay
  bit-identical to vanilla) and `w → 1` on genuinely novel local environments.
- **`V_phys`** — a robust short-range physical repulsion (extended ZBL /
  Born–Mayer, or a per-pair physics residual) calibrated cheaply from dimer scans.

ZBL is recovered as the special case `novelty → 1` as `r → 0`. The correction is
**additive at the output and self-nulling at equilibrium**, so accuracy on the
in-distribution set is preserved by construction while distorted structures are
pulled back toward the correct physics.

## Key results (MACE-MH-0 base)

| Dataset | vanilla F R² | naive kNN gate | **RND + SelectiveNet gate** |
|---|---|---|---|
| u200 (clean target) | 0.998 | −2.03 | **0.998** (unchanged) |
| MPtrj (in-distribution base) | 0.986 | −3.03 | **0.986** (unchanged) |
| keep_test (compressed) | 0.650 | 0.728 | **0.738** (F MAE −7.8%) |

A naive latent-distance (kNN) gate *destroys* the base through false positives
(it fired on 93 % of base atoms); the RND gate separates compressed environments
from the base by ~10⁴× in novelty magnitude, giving a formal risk–coverage
guarantee. The method also improves high-pressure equations of state, MD
short-range stability, and radiation-damage observables, and transfers across
architectures (MACE, DPA-3.1, and — for comparison — M3GNet / CHGNet).

## Repository layout

```
ZBL_anchored/
├── anchor/                 # the method on top of MACE (core)
│   ├── scripts/            #   RND/rho gates, pair-physics, calibration, tuning, eval, figures
│   ├── md_stability/       #   AnchorCalculator + MD short-range stability suite
│   └── raddmg/             #   radiation-damage validation (defects, recoil, SrTiO3, Wigner–Seitz)
├── dpa_variant/            # the same method on top of DPA-3.1
├── mace_zbl_training/      # ZBL-MACE fine-tuning / evaluation pipeline (01…06)
├── lammps/                 # deploying the potentials in LAMMPS via fix external
│   ├── scripts/            #   mlip_fixext.py driver + profilers + build/convert utilities
│   ├── mace_smoke/ dpa_smoke/  numerical reference smoke tests
│   └── examples/           #   ready-to-edit input decks
├── benchmarks/             # MACE acceleration benchmarks (fp32 / cuEquivariance / torch.compile)
├── patches/                # bit-identical safe-optimisation patch for MACE
├── figures/                # poster + headline figure panels + figure-generation code
├── config.example.env      # every external path is an env var — copy to config.env and fill in
└── environment.yml
```

Each sub-directory has its own README with run instructions.

## What is *not* in this repository

By design the repo contains **code only**. The following are excluded (see
`.gitignore`) and must be supplied by the user:

- **Foundation model weights** (MACE-MH-0/1, DPA-3.1). These are governed by the
  upstream model licenses (e.g. the ACEsuit models are released under a
  non-commercial academic license) and are **not** redistributed here. Obtain them
  from the upstream projects — see `mace_zbl_training/scripts/01_download_models.py`
  and the ACEsuit / DeepModeling releases.
- **Evaluation datasets** (VASP splits, MPtrj subsets). Point the scripts at your
  own copies via the environment variables in `config.example.env`.
- **Generated artifacts** (`rnd.pt`, `rho_reference.npz`, dimer tables) — build
  them locally with the scripts in `anchor/scripts/`.
- Large run outputs, logs, and third-party libraries.

## Setup

```bash
git clone <this-repo> ZBL_anchored && cd ZBL_anchored
conda env create -f environment.yml    # or use your own env with ASE/MACE/torch
cp config.example.env config.env        # then edit config.env with paths on your machine
source config.env
```

No script hard-codes a machine-specific path; every external resource is read from
an environment variable documented in `config.example.env`.

## License

The code in this repository is released under the **MIT License** (see `LICENSE`).
Foundation model weights, third-party libraries (MACE, DPA/DeePMD-kit,
cuEquivariance, e3nn, ASE, …) and any datasets retain their own licenses and are
not covered by this repository's license.

## Citation / provenance

This is research code accompanying the "uncertainty-gated physical anchor"
(ZBL-anchored) study. The `patches/mace_safe_opts.patch` applies against
ACEsuit/mace commit `4d2da09`. If you use this work, please also cite the
underlying MACE, DPA-3.1, RND, and SelectiveNet references.
