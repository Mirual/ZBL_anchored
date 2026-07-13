# figures/ — poster and headline figures

Curated images for the ZBL-anchored study, kept separate from the code.

```
figures/
├── poster/                 # the A0 conference poster (PNG + PDF)
├── panels/                 # the individual poster figure panels
├── anch_prez.png           # hero illustration (README header)
├── method_scheme.png       # method flow diagram (README)
├── anchor_zones.png        # where the anchor acts: V(r) distance zones (README)
├── radiation_cascade.png   # why short range matters: cascade explainer (README)
├── make_method_scheme.py   # regenerates method_scheme.png
├── make_zones_figure.py    # regenerates anchor_zones.png
├── make_cascade_figure.py  # regenerates radiation_cascade.png
└── poster_figures.py       # code that regenerates the poster panels
```

- `poster/physanchor_poster_A0.png` / `.pdf` — the full poster.
- `panels/poster_gate.png` — RND gate separation (novelty vs distance).
- `panels/poster_highP.png` — high-pressure equation-of-state improvement.
- `panels/poster_md.png` — MD short-range stability.
- `panels/poster_crossmodel.png` — cross-architecture transfer (MACE / DPA / …).
- `panels/poster_vs_ft.png` — anchor vs fine-tuning comparison.

`poster_figures.py` regenerates the panels from the study's result artifacts
(which are not committed — see the per-directory READMEs for how to produce them).
All other intermediate plots in the project are reproducible from the
`make_*_figures.py` scripts under `../anchor/scripts/`.
