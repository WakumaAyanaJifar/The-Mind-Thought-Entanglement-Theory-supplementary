# Paper: Section 3.3 'What the identity tests' and Supplementary S2. Expected: 1125/1125 excitatory two-fibre models satisfy G1 and G3; 1445/1445 mixed-sign models have one negative weight.
# Part of: the Mind-Thought Entanglement (MTE) framework - classical two-field model. Run via ../run_all.py
"""Conventional competitor: one excitatory field, two axonal systems (short intracortical,
long corticocortical), Nunez/Jirsa-Haken type. Linearized:
 (d_t^2 + 2 G_i d_t + G_i^2 - v_i^2 Lap) psi_i = (G_i^2 + G_i d_t) w_i (psi_1+psi_2 + I)
Observed field: psi = psi_1 + psi_2.  Check G1 (Vieta), G3 (positive weights)."""
import numpy as np
rng=np.random.default_rng(1)
def model(G,v,w):
    # returns dispersion D(omega,k) coefficients as function, and static response chi(q), q=k^2
    def L(i,om,q): return -om**2-2j*G[i]*om+G[i]**2+v[i]**2*q
    def W(i,om): return (G[i]**2-1j*G[i]*om)*w[i]
    def D(om,q): return (L(0,om,q)-W(0,om))*(L(1,om,q)-W(1,om))-W(0,om)*W(1,om)
    def N(om,q): return W(0,om)*L(1,om,q)+W(1,om)*L(0,om,q)   # chi = N/D
    return D,N
ok=0; trials=2000; fails=[]
for t in range(trials):
    G=rng.uniform(0.3,3,2); v=rng.uniform(0.3,3,2); w=rng.uniform(0.05,0.9,2)*np.array([1,1])
    D,N=model(G,v,w)
    # static: D(0,q) quadratic in q
    qs=np.linspace(-50,50,5)
    cq=np.polyfit(qs,[D(0,q).real for q in qs],2)
    if cq[2]<=0: continue   # need stable uniform state: D(0,0)>0
    roots=np.roots(cq)
    if np.any(np.abs(roots.imag)>1e-9) or np.any(roots.real>=0): continue
    kap2=np.sort(-roots.real)            # kappa^2
    # residues of chi(q)=N/D at q=-kappa^2
    cN=np.polyfit(qs,[N(0,q).real for q in qs],1)
    R=[np.polyval(cN,-k2)/np.polyval(np.polyder(cq),-k2) for k2 in kap2]
    R=np.array(R)/sum(R)
    # G1: product of the 4 k=0 frequency roots
    oms=np.linspace(-3,3,7)
    co=np.polyfit(oms,[D(o,0) for o in oms],4)   # quartic in omega
    wr=np.roots(co)
    lhs=np.sqrt(kap2.prod()); rhs=np.sqrt(np.prod(np.abs(wr)))/(v[0]*v[1])
    g1ok=abs(lhs-rhs)/lhs<1e-6; g3ok=np.all(R>0)
    ok+=g1ok and g3ok
    if not (g1ok and g3ok): fails.append((G,v,w,R,lhs,rhs))
    trials_used=t
print("valid cases passing G1 and G3:",ok,"failures:",len(fails))
# E-I sign version: w2 negative (inhibitory second system)
cnt=0;neg=0
for t in range(2000):
    G=rng.uniform(0.3,3,2); v=rng.uniform(0.3,3,2); w=np.array([rng.uniform(0.05,0.9),-rng.uniform(0.05,0.9)])
    D,N=model(G,v,w)
    qs=np.linspace(-50,50,5); cq=np.polyfit(qs,[D(0,q).real for q in qs],2)
    if cq[2]<=0: continue
    roots=np.roots(cq)
    if np.any(np.abs(roots.imag)>1e-9) or np.any(roots.real>=0): continue
    kap2=np.sort(-roots.real); cN=np.polyfit(qs,[N(0,q).real for q in qs],1)
    R=np.array([np.polyval(cN,-k2)/np.polyval(np.polyder(cq),-k2) for k2 in kap2])
    cnt+=1; neg+= np.any(R<0)
print("mixed-sign cases:",cnt," with a negative weight:",neg)
