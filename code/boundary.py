# Paper: Section 3.4, Fig. 3 and Supplementary S4. Expected: LOC slopes -0.504/-0.506/-0.252, ROC -1.000/-1.000/-0.500, peak ratio = 1/[2(1+sqrt(W/delta))], post-jump ratio 4, bias crossover sqrt(12 b J).
# Part of: the Mind-Thought Entanglement (MTE) framework - classical two-field model. Run via ../run_all.py
"""Uniform sector with cross term b != 0; full two-field linearization (g2 retained).
Control: r = mu^2 - C (anaesthetic dose raises r). Checks exponents and the parameter-free ratio."""
import numpy as np
m=2.0; lam=1.0; g1=2.5; g2=1.2; c1=1.0; c2=0.7; G1=0.6; G2=0.9
m2=m*m; C=g1**2/m2; b=g1*g2/(2*m2); leff=lam-g2**2/(2*m2); W=9*b**2/(4*leff)
def branches(r, J=0.0):
    # stationary points of V' = r P - 3b P^2 + leff P^3 - J
    rts=np.roots([leff,-3*b,r,-J]); return np.sort(rts[np.abs(rts.imag)<1e-10].real)
def lin(P, r):
    mu2=r+C; phi=(g1*P+g2*P**2/2)/m2
    a=mu2+3*lam*P**2-g2*phi; ge=g1+g2*P
    Vpp=r-6*b*P+3*leff*P**2
    chi0=1/(a-ge**2/m2)
    # slowest relaxation time from k=0 quartic
    q=np.polymul([-1,-1j*G1,a],[-1,-1j*G2,m2]); q[-1]-=ge**2
    w=np.roots(q); tau=1/np.min(-w.imag)
    # static poles (a - c1^2 x)(m2 - c2^2 x) = ge^2
    s=np.polymul([-c1**2,a],[-c2**2,m2]); s[-1]-=ge**2
    x=np.sort(np.roots(s).real); xi=1/np.sqrt(x[0])
    return chi0, 1/Vpp, tau, xi
print(f"C={C:.4f} b={b:.4f} leff={leff:.4f} W=9b^2/(4leff)={W:.4f}")
# Schur check at a random ordered point
r=0.1; P=branches(r)[-1]; c,cv,_,_=lin(P,r); print("chi from linearization vs 1/V'':",c,cv)
# exponents
d=np.logspace(-6,-3,8)*W
def slope(y): return np.polyfit(np.log(d),np.log(y),1)[0]
LOC=np.array([lin(branches(W-dd)[-1],W-dd) for dd in d])   # ordered branch approaching saddle-node
ROC=np.array([lin(0.0,dd) for dd in d])                     # symmetric state approaching r=0 from above
print("LOC slopes chi,tau,xi:",[round(slope(LOC[:,i]),3) for i in (0,2,3)])
print("ROC slopes chi,tau,xi:",[round(slope(ROC[:,i]),3) for i in (0,2,3)])
# parameter-free ratio chi_LOC/chi_ROC at equal distance vs 1/[2(1+sqrt(W/delta))]
for dd in [0.01*W,0.1*W,0.5*W,W]:
    cl=lin(branches(W-dd)[-1],W-dd)[0]; cr=lin(0.0,dd)[0]
    print(f"delta/W={dd/W:5.2f} ratio numeric={cl/cr:.6f} formula={1/(2*(1+np.sqrt(W/dd))):.6f}")
# post-jump values: after LOC (P=0 at r=W) vs after ROC (ordered at r=0)
aL=lin(0.0,W)[0]; aR=lin(branches(0.0)[-1],0.0)[0]; print("post-jump chi ratio (after LOC)/(after ROC):",aL/aR)
# J crossover at ROC: exponent 1 -> 1/2 below delta_J ~ sqrt(12 b J)
J=1e-4; dJ=np.sqrt(12*b*J); print("delta_J=",dJ)
def rocJ(dd):
    # r_sn for disordered-branch with J: follow smallest-|P| root until it disappears
    return None

# --- crossover check at ROC with a small bias J (b J > 0) ---
J=1e-4
rs=np.linspace(0.3,0.0,300001)
prev=None; rsn=None
for r in rs:
    br=branches(r,J)
    small=br[np.argmin(np.abs(br))] if len(br) else None
    if len(br)<3 and prev is not None and r<0.1: rsn=r; break
    prev=small
print("numerical ROC saddle-node with J:",rsn," sqrt(12 b J):",np.sqrt(12*b*J))
def chiJ(r):
    br=branches(r,J); P=br[np.argmin(np.abs(br))]; return 1/(r-6*b*P+3*leff*P**2)
for lo,hi,lab in [(1e-4,1e-3,'near (delta<<delta_J)'),(0.1,0.3,'far (delta>>delta_J)')]:
    dd=np.logspace(np.log10(lo*(rsn if lab.startswith('near') else 1)),np.log10(hi*(rsn if lab.startswith('near') else 1)),6)
    y=[chiJ(rsn+x) for x in dd] if lab.startswith('near') else [chiJ(x) for x in dd]
    print(lab,"slope:",round(np.polyfit(np.log(dd),np.log(y),1)[0],3))
