import json, numpy as np
from pathlib import Path
R=Path("results")
def met(f):
    frames=json.loads(Path(f).read_text())["frames"]
    Er,Ep,nat,Fr,Fp=[],[],[],[],[]
    for fr in frames:
        if fr["E_pred"] is None: continue
        Er.append(fr["E_ref"]); Ep.append(fr["E_pred"]); nat.append(fr["n_atoms"])
        if fr["F_ref"] is not None and fr["F_pred"] is not None:
            Fr.append(np.array(fr["F_ref"]).reshape(-1)); Fp.append(np.array(fr["F_pred"]).reshape(-1))
    Er,Ep,nat=np.array(Er),np.array(Ep),np.array(nat)
    eE=np.abs((Ep-Er)/nat).mean()*1000
    eR2=1-((Er-Ep)**2).sum()/((Er-Er.mean())**2).sum()
    Fr=np.concatenate(Fr); Fp=np.concatenate(Fp)
    fM=np.abs(Fp-Fr).mean(); fR2=1-((Fr-Fp)**2).sum()/((Fr-Fr.mean())**2).sum()
    return eE,eR2,fM,fR2,len(Er)
ds=[("weakly distorted (target)","u200"),("distorted (compressed OOD)","keep"),
    ("distorted (full set)","keepfull"),("MPtrj (baseline)","mptrj")]
print(f"{'dataset':22s} {'model':7s} {'n':>5s} | {'E MAE meV/at':>12s} {'E R²':>7s} | {'F MAE eV/Å':>10s} {'F R²':>7s}")
print("-"*82)
for name,tag in ds:
    va,an=R/f"all_vanilla_{tag}.json", R/f"all_anchor_{tag}.json"
    if not (va.exists() and an.exists()):
        print(f"{name:22s} (pending: {'vanilla' if not va.exists() else ''} {'anchor' if not an.exists() else ''})\n"); continue
    for lab,p in [("vanilla",va),("anchor",an)]:
        eE,eR2,fM,fR2,n=met(p)
        print(f"{name:22s} {lab:7s} {n:5d} | {eE:12.1f} {eR2:7.3f} | {fM:10.3f} {fR2:7.3f}")
    print()
