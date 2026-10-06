"""Paper: Section 3.4 (window of Eq. (24)) and Supplementary S4, Table S5.
Stochastic simulation of the boundary on a noisy two-dimensional sheet. The slow mode near the
transitions obeys relaxational dynamics in V_eff (Fig. 3 parameters):
    dPsi = [c^2 lap Psi - V'(Psi)] dt + sqrt(2 D) dW,  V = r Psi^2/2 - b Psi^3 + l Psi^4/4,
on a periodic N x N lattice (spacing a), integrated with an exponential (spectral) step for the
gradient term. The integrated susceptibility is estimated from the fluctuation-dissipation
relation, chi = A Var(mean Psi)/D (A = sheet area), and compared with the mean-field 1/V''.
Two noise levels are set by r_G/W = 0.01 and 0.1 (Eq. (23))."""
import numpy as np, sys
b,l,c2=0.375,0.82,1.3; W=9*b*b/(4*l); N=64; a=0.5; A=(N*a)**2; Lam=np.pi/a
k=2*np.pi*np.fft.fftfreq(N,a); K2=np.add.outer(k**2,k**2)
rng=np.random.default_rng(11)
def D_for(rG):
    return rG*4*np.pi*c2/(3*l*np.log(c2*Lam**2/rG))
def run(r,P0,D,ttot,dt=0.25):
    E=np.exp(-c2*K2*dt); P=np.full((N,N),P0); ms=[]; esc=False
    ns=int(ttot/dt); burn=ns//10; sd=np.sqrt(2*D*dt)/a
    Ps=(3*b-2*np.sqrt(l*max(W-r,0)))/(2*l) if r<W else None   # saddle between the two states
    for n in range(ns):
        P=P-dt*(r*P-3*b*P**2+l*P**3)+sd*rng.standard_normal((N,N))
        P=np.real(np.fft.ifft2(np.fft.fft2(P)*E))
        if n>=burn and n%4==0:
            m=P.mean(); ms.append(m)
            if Ps is not None and ((P0>0 and m<Ps) or (P0==0 and Ps>0 and m>Ps)): esc=True; break
    ms=np.array(ms)
    if esc or len(ms)<40: return np.nan, esc, 0.0
    bm=np.array([x.var() for x in np.array_split(ms,8)]); return A*ms.var()/D, esc, A*bm.std()/np.sqrt(8)/D
quick='--quick' in sys.argv
dW=[0.003,0.01,0.03,0.1,0.3,1.0]
print(f"W={W:.4f}, lattice {N}x{N}, a={a}, area {A:.0f}")
for rGW in [0.01,0.1]:
    D=D_for(rGW*W); print(f"\nr_G/W={rGW}: D={D:.4g}")
    print(" delta/W | LOC chi_sim/chi_MF | ROC chi_sim/chi_MF")
    for x in dW:
        d=x*W
        VL=2*np.sqrt(W*d)+2*d; PL=(3*b+2*np.sqrt(l*d))/(2*l); VR=d
        tL=min(200/VL,6e4); tR=min(200/VR,6e4)
        if quick: tL/=4; tR/=4
        if x<1.0: cL,eL,sL=run(W-d,PL,D,tL)
        else: cL,eL,sL=np.nan,False,0.0
        cR,eR,sR=run(d,0.0,D,tR)
        f=lambda c,s_,V,e: 'escaped' if e else ('   --  ' if np.isnan(c) else f"{c*V:5.2f}+-{s_*V:4.2f}")
        print(f"  {x:6.3f} |   {f(cL,sL,VL,eL):12s} |   {f(cR,sR,VR,eR):12s}",flush=True)
