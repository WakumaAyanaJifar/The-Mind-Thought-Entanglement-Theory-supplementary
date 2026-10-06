"""Paper: Supplementary S5 (power analysis for the exponent test E1).
Each simulated participant contributes the spatially integrated response at 6 dose steps on
each side, geometrically spaced in delta/W over [0.02, 0.3] (LOC) and [0.1, 1.0] (ROC), the
budgeted windows of Section 3.4. True values follow the exact forms (J = 0); the measured log
response has Gaussian error with SD equal to the coefficient of variation CV per step; the
position of each transition is mis-estimated by a common offset with SD = half a step
(0.01 W). Per participant the local exponents are fitted by log-log regression on each side;
the group test is a one-sided one-sample t-test that nu_ROC - nu_LOC > 0 at alpha = 0.05.
Also reported: the expected difference when background drive is large (delta_J = 0.3 W), the
case the J1 arm must exclude."""
import numpy as np
from scipy import stats
rng=np.random.default_rng(2026)
W=1.0; dL=np.geomspace(0.02,0.3,6)*W; dR=np.geomspace(0.1,1.0,6)*W
chiL=lambda d: 1/(2*(np.sqrt(W*d)+d)); chiR=lambda d: 1/d
def chiR_J(d,dJ):            # ROC as a saddle-node at delta_J (bJ>0): local form 1/(2 sqrt(dJ (d)) + ...)
    return 1/(2*(np.sqrt(dJ*d)+d))
def participant(cv,dJ=None):
    oL=rng.normal(0,0.01*W); oR=rng.normal(0,0.01*W)
    yL=np.log(chiL(np.clip(dL+oL,1e-4,None)))+rng.normal(0,cv,6)
    yR=np.log((chiR(np.clip(dR+oR,1e-4,None)) if dJ is None else chiR_J(np.clip(dR+oR,1e-4,None),dJ)))+rng.normal(0,cv,6)
    nL=-np.polyfit(np.log(dL),yL,1)[0]; nR=-np.polyfit(np.log(dR),yR,1)[0]; return nR-nL
print("expected local exponents in these windows: LOC %.3f, ROC %.3f (difference %.3f)"%(
    -np.polyfit(np.log(dL),np.log(chiL(dL)),1)[0], -np.polyfit(np.log(dR),np.log(chiR(dR)),1)[0],
    -np.polyfit(np.log(dR),np.log(chiR(dR)),1)[0]+np.polyfit(np.log(dL),np.log(chiL(dL)),1)[0]))
print("with strong background drive (delta_J=0.3W) the expected difference falls to %.3f"%(
    -np.polyfit(np.log(dR),np.log(chiR_J(dR,0.3)),1)[0]+np.polyfit(np.log(dL),np.log(chiL(dL)),1)[0]))
print("power (2000 simulated studies each):")
print("  CV per step | n=5   n=8   n=12  n=20")
for cv in [0.1,0.2,0.3,0.5]:
    row=[]
    for n in [5,8,12,20]:
        p=np.mean([stats.ttest_1samp([participant(cv) for _ in range(n)],0,alternative='greater').pvalue<0.05 for _ in range(2000)])
        row.append(p)
    print(f"  {cv:10.1f}  | "+"  ".join(f"{p:.2f}" for p in row),flush=True)
