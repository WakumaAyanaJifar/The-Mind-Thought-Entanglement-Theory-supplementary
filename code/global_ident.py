"""Paper: Section 3.8 and Supplementary S6 (global identifiability).
Multi-start search over a wide box for parameter sets that reproduce the transient response
chi(omega,k) of Fig. 2b on the grid of recovery7.py (h = 2 mm). Starts are log-uniform within a
factor 10 of the truth (bounds: factor 30). Any converged fit whose relative misfit is below
1e-6 but whose parameters differ from the truth by more than 1% would be a distinct
indistinguishable parameter set. Also: profile of the misfit along geff^2 (re-optimizing the rest)."""
import numpy as np
from scipy.optimize import least_squares
rng=np.random.default_rng(3)
names=['A','c1','c2','g1','g2','a','m2','gg']
th0=np.array([1.0,1.0,0.7,0.6,0.9,2.6875,4.0,6.25])
def chi(th,W,K):
    A,c1,c2,G1,G2,a,m2,g=th; num=c2**2*K**2+m2-W**2-1j*G2*W
    return A*num/((c1**2*K**2+a-W**2-1j*G1*W)*num-g)
k=np.linspace(2*np.pi/12,np.pi/0.2,40); w=np.linspace(0.05,2*np.pi,120); K,Wg=np.meshgrid(k,w)
y=chi(th0,Wg,K); sc=np.sqrt(np.mean(np.abs(y)**2))
def res(p,fix=None):
    th=np.exp(p) if fix is None else np.exp(np.insert(p,7,fix))
    d=(chi(th,Wg,K)-y)/sc; return np.concatenate([d.real.ravel(),d.imag.ravel()])
lb=np.log(th0/30); ub=np.log(th0*30)
sols=[]; n=200
for i in range(n):
    p0=np.log(th0)+rng.uniform(-np.log(10),np.log(10),8)
    r=least_squares(res,p0,bounds=(lb,ub),max_nfev=4000,xtol=1e-12,ftol=1e-14)
    rel=np.sqrt(np.mean(r.fun**2)); sols.append((rel,np.exp(r.x)))
rels=np.array([s[0] for s in sols]); dev=np.array([np.max(np.abs(s[1]/th0-1)) for s in sols])
fit=rels<1e-6
print(f"{n} starts: {fit.sum()} reach misfit < 1e-6; of these, max parameter deviation from truth = {dev[fit].max():.2e}")
print(f"other local minima: {np.sum(~fit)}; smallest misfit among them = {rels[~fit].min() if (~fit).any() else float('nan'):.3e} (relative RMS)")
print(f"distinct indistinguishable sets (misfit<1e-6, deviation>1%): {np.sum(fit & (dev>0.01))}")
print("profile along geff^2 (others re-optimized from the truth): relative RMS misfit")
for f in [0.9,0.95,0.99,1.01,1.05,1.1]:
    p0=np.delete(np.log(th0),7)
    r=least_squares(lambda p: res(p,fix=np.log(6.25*f)),p0,bounds=(np.delete(lb,7),np.delete(ub,7)),max_nfev=4000)
    print(f"   geff^2 x{f}: {np.sqrt(np.mean(r.fun**2)):.3e}")
