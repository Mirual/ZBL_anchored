#!/usr/bin/env python3
"""Metrics: vanilla DPA-3.1 vs RND-gated pairphys anchor (CLAUDE.md pct_compare contract).

Per condition × {vanilla, anchor}: F R², F MAE (eV/Å), E MAE (eV, eV/atom), E mean %, F mean %.
Headline = keep F R² (improvement) + MPtrj/u200 F R² (base preservation). Parity figure of forces.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"
CONDS = ["keep_test", "u200_test", "mptrj"]


def r2(p, r):
    p, r = np.asarray(p), np.asarray(r)
    ss = ((r - r.mean()) ** 2).sum()
    return 1.0 - ((p - r) ** 2).sum() / ss if ss > 0 else float("nan")


def metrics(path):
    fr = json.loads(Path(path).read_text())["frames"]
    Ep, Er, Emae_pa, Epct = [], [], [], []
    Fp, Fr, Fpct = [], [], []
    for f in fr:
        if f["E_pred"] is None or f["F_pred"] is None:
            continue
        Ep.append(f["E_pred"]); Er.append(f["E_ref"])
        Emae_pa.append(abs(f["E_pred"] - f["E_ref"]) / f["n_atoms"])
        if abs(f["E_ref"]) > 0:
            Epct.append(abs(f["E_pred"] - f["E_ref"]) / abs(f["E_ref"]) * 100)
        fp = np.asarray(f["F_pred"]); rf = np.asarray(f["F_ref"])
        Fp.append(fp.reshape(-1)); Fr.append(rf.reshape(-1))
        n = np.linalg.norm(rf, axis=1); m = n > 0
        Fpct.append(np.linalg.norm(fp - rf, axis=1)[m] / n[m] * 100)
    Ep, Er = np.array(Ep), np.array(Er)
    Fp, Fr = np.concatenate(Fp), np.concatenate(Fr)
    return dict(n=len(Ep),
                E_mae=np.abs(Ep - Er).mean(), E_mae_pa=np.mean(Emae_pa), E_pct=np.mean(Epct), E_r2=r2(Ep, Er),
                F_mae=np.abs(Fp - Fr).mean(), F_pct=np.mean(np.concatenate(Fpct)), F_r2=r2(Fp, Fr),
                Fp=Fp, Fr=Fr)


def main():
    rows = {}
    print(f"{'cond':>10s} {'model':>8s} | {'E MAE':>8s} {'E/atom':>8s} {'E mean%':>8s} {'E R²':>7s} | "
          f"{'F MAE':>7s} {'F mean%':>8s} {'F R²':>7s}")
    print("-" * 90)
    for cond in CONDS:
        for tag in ("vanilla", "zbl", "anchor"):
            p = RES / f"{tag}_{cond}.json"
            if not p.exists():
                continue
            m = metrics(p); rows[(cond, tag)] = m
            print(f"{cond:>10s} {tag:>8s} | {m['E_mae']:8.3f} {m['E_mae_pa']:8.4f} {m['E_pct']:8.2f} "
                  f"{m['E_r2']:7.3f} | {m['F_mae']:7.4f} {m['F_pct']:8.1f} {m['F_r2']:7.3f}")
        print()

    # headline
    if ("keep_test", "vanilla") in rows and ("keep_test", "anchor") in rows:
        v, a = rows[("keep_test", "vanilla")], rows[("keep_test", "anchor")]
        print(f"HEADLINE keep F R²: {v['F_r2']:.3f} → {a['F_r2']:.3f}   "
              f"F MAE {v['F_mae']:.4f} → {a['F_mae']:.4f} ({(a['F_mae']/v['F_mae']-1)*100:+.1f}%)")
    if ("mptrj", "vanilla") in rows and ("mptrj", "anchor") in rows:
        v, a = rows[("mptrj", "vanilla")], rows[("mptrj", "anchor")]
        print(f"BASE     MPtrj F R²: {v['F_r2']:.3f} → {a['F_r2']:.3f}  (should stay ≈)")

    # parity figure: forces, rows {keep,mptrj} × cols {vanilla,anchor}
    FIG.mkdir(exist_ok=True)
    tags = ["vanilla", "zbl", "anchor"]
    colors = {"vanilla": "#7f8c8d", "zbl": "#2980b9", "anchor": "#27ae60"}
    panels = [(c, t) for c in ("keep_test", "mptrj") for t in tags]
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))
    for k, (cond, tag) in enumerate(panels):
        a = ax[k // 3][k % 3]
        if (cond, tag) not in rows:
            a.set_visible(False); continue
        m = rows[(cond, tag)]
        lim = np.percentile(np.abs(m["Fr"]), 99.5)
        a.scatter(m["Fr"], m["Fp"], s=2, alpha=0.25, color=colors.get(tag, "#7f8c8d"))
        a.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_title(f"{cond} · {tag}\nF R²={m['F_r2']:.3f}  MAE={m['F_mae']:.3f} eV/Å", fontsize=10,
                    fontweight="bold" if cond == "keep_test" else "normal")
        a.set_xlabel("F_ref (eV/Å)"); a.set_ylabel("F_pred (eV/Å)")
    fig.suptitle("DPA-3.1: vanilla vs pure-ZBL vs RND-gated anchor — forces parity", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "dpa_anchor_parity.png", dpi=130)
    print(f"\nsaved {FIG/'dpa_anchor_parity.png'}")

    # machine-readable summary
    summ = {f"{c}/{t}": {k: float(v) for k, v in rows[(c, t)].items() if k != "Fp" and k != "Fr"}
            for (c, t) in rows}
    (RES / "compare_summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
