# Paper: Section 3.8 and Supplementary Table S3. Expected: 0 failed fits in all 9 conditions; median errors as in Table S3 (fixed random seed).
# Part of: the Mind-Thought Entanglement (MTE) framework - classical two-field model. Run via ../run_all.py
"""k-space emulation of the transient (P1) route: recovery errors of the seven dispersion
parameters (c1,c2,gamma1,gamma2,a,m^2,geff^2) plus amplitude A, from noisy chi(omega,k).
Grid as in fisher.py (40 k-bins up to pi/h, 120 frequency bins); complex white noise at a
per-bin SNR; least-squares fit in log-parameters from starts perturbed by U(0.6,1.6),
bounded within a factor 10 of the truth; 40 realizations per condition."""
import numpy as np
from scipy.optimize import least_squares
names=['A','c1','c2','gamma1','gamma2','a','m2','geff2']
th0=np.array([1.0,1.0,0.7,0.6,0.9,2.6875,4.0,6.25])
def chi(th,W,K):
    A,c1,c2,G1,G2,a,m2,g=th
    num=c2**2*K**2+m2-W**2-1j*G2*W
    return A*num/((c1**2*K**2+a-W**2-1j*G1*W)*num-g)
rng=np.random.default_rng(7)
out={}
for h in [0.2,0.5,1.0]:
    k=np.linspace(2*np.pi/12,np.pi/h,40); w=np.linspace(0.05,2*np.pi,120); K,Wg=np.meshgrid(k,w)
    f0=chi(th0,Wg,K); rms=np.sqrt(np.mean(np.abs(f0)**2))
    for snr in [5,10,20]:
        sig=rms/snr; errs=[]; fails=0
        for rep in range(40):
            y=f0+sig/np.sqrt(2)*(rng.standard_normal(f0.shape)+1j*rng.standard_normal(f0.shape))
            def res(p):
                d=chi(np.exp(p),Wg,K)-y; return np.concatenate([d.real.ravel(),d.imag.ravel()])/sig
            best=None
            for s in range(3):
                p0=np.log(th0*rng.uniform(0.6,1.6,8))
                try:
                    r=least_squares(res,p0,bounds=(np.log(th0/10),np.log(th0*10)),x_scale=1.0,max_nfev=4000)
                    if best is None or r.cost<best.cost: best=r
                except Exception: pass
            if best is None or not best.success: fails+=1; continue
            errs.append(np.abs(np.exp(best.x)/th0-1))
        e=np.array(errs)
        out[(h,snr)]=(np.median(e,0),np.percentile(e,90,0),fails)
        print(f"h={h*10:.0f}mm SNR/bin={snr}: fails={fails} median%:",' '.join(f"{n}={100*v:.2f}" for n,v in zip(names,np.median(e,0))))
        print("      90th%:",' '.join(f"{n}={100*v:.2f}" for n,v in zip(names,np.percentile(e,90,0))))
np.save('recovery7.npy',out,allow_pickle=True)
