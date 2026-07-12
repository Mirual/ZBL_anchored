import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path("results"); FIG=Path("figures")
def met(f):
    fr=json.loads((R/f).read_text())["frames"]
    Er,Ep,nat,Fr,Fp=[],[],[],[],[]
    for x in fr:
        if x["E_pred"] is None: continue
        Er.append(x["E_ref"]); Ep.append(x["E_pred"]); nat.append(x["n_atoms"])
        if x["F_ref"] and x["F_pred"]:
            Fr.append(np.array(x["F_ref"]).reshape(-1)); Fp.append(np.array(x["F_pred"]).reshape(-1))
    Er,Ep,nat=map(np.array,(Er,Ep,nat)); Fr,Fp=np.concatenate(Fr),np.concatenate(Fp)
    return dict(fMAE=np.abs(Fp-Fr).mean(), fR2=1-((Fr-Fp)**2).sum()/((Fr-Fr.mean())**2).sum())
DS=[("u200\ntarget","u200"),("keep_test\ncompressed","keep"),("keep_full\ncompressed","keepfull"),("MPtrj\nbase","mptrj")]
mods=[("vanilla","all_vanilla","#34495e"),("kNN-ρ anchor","all_anchor","#c0392b"),("RND anchor","rnd","#27ae60")]
x=np.arange(len(DS)); w=0.26
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
for mi,(lab,pre,col) in enumerate(mods):
    fr2=[met(f"{pre}_{t}.json")["fR2"] for _,t in DS]
    fma=[met(f"{pre}_{t}.json")["fMAE"] for _,t in DS]
    ax[0].bar(x+(mi-1)*w, np.clip(fr2,-3.5,None), w, label=lab, color=col, edgecolor="k", lw=.5)
    ax[1].bar(x+(mi-1)*w, fma, w, label=lab, color=col, edgecolor="k", lw=.5)
    for xi,v in enumerate(fr2): ax[0].annotate(f"{v:.2f}",(xi+(mi-1)*w,max(v,-3.4)),ha="center",va="bottom" if v>=0 else "top",fontsize=6.5)
ax[0].set_title("Force R²  (↑ better; <0 = base destroyed)",fontweight="bold"); ax[0].set_ylim(-3.6,1.15)
ax[0].axhline(0,color="k",lw=.6); ax[0].set_ylabel("F R²")
ax[1].set_title("Force MAE  (↓ better)",fontweight="bold"); ax[1].set_yscale("symlog"); ax[1].set_ylabel("eV/Å")
for a in ax:
    a.set_xticks(x); a.set_xticklabels([d for d,_ in DS],fontsize=8); a.legend(fontsize=8); a.grid(axis="y",alpha=.3)
fig.suptitle("RND-novelty gate vs kNN-ρ gate — held-out, full datasets (MACE-MH-0)",fontsize=13,fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"rnd_vs_knn.png",dpi=130); print("saved figures/rnd_vs_knn.png")
