# Paper: Section 3.8 and Supplementary Table S2. Expected: rank 8/8; condition numbers 36.1, 37.0, 31.2; Cramer-Rao bounds as in Table S2.
# Part of: the Mind-Thought Entanglement (MTE) framework - classical two-field model. Run via ../run_all.py
"""Local identifiability of the linear sector from the transient spectrum (P1).
Observable: chi(w,k) = A (c2^2 k^2 + m^2 - w^2 - i g2 w)/D(w,k). Jacobian w.r.t. log-parameters."""
import numpy as np
names=['A','c1','c2','gamma1','gamma2','a','m^2','geff^2']
th0=np.array([1.0,1.0,0.7,0.6,0.9,2.6875,4.0,6.25])
def chi(th,W,K):
    A,c1,c2,G1,G2,a,m2,g=th
    num=c2**2*K**2+m2-W**2-1j*G2*W
    D=(c1**2*K**2+a-W**2-1j*G1*W)*num-g
    return A*num/D
def analyse(kmax,label,snr_bin=10.0):
    k=np.linspace(2*np.pi/12,kmax,40); w=np.linspace(0.05,2*np.pi,120)   # 0.8..100 Hz with 10 ms unit
    K,Wg=np.meshgrid(k,w)
    f0=chi(th0,Wg,K); sig=np.sqrt(np.mean(np.abs(f0)**2))/snr_bin
    J=[]
    for i in range(len(th0)):
        h=1e-6; tp=th0.copy(); tp[i]*=np.exp(h); tm=th0.copy(); tm[i]*=np.exp(-h)
        d=(chi(tp,Wg,K)-chi(tm,Wg,K))/(2*h)
        J.append(np.concatenate([d.real.ravel(),d.imag.ravel()]))
    J=np.array(J).T
    s=np.linalg.svd(J,compute_uv=False)
    F=J.T@J/sig**2; crlb=np.sqrt(np.diag(np.linalg.inv(F)))   # relative (log) std
    print(f"{label}: rank {np.linalg.matrix_rank(J,tol=s[0]*1e-10)}/8, cond {s[0]/s[-1]:.1f}")
    print("   CRLB relative SD (%):",", ".join(f"{n}={100*c:.3f}" for n,c in zip(names,crlb)))
    return s
analyse(np.pi/0.2,"h=2 mm (k up to 15.7 /cm)")
analyse(np.pi/0.5,"h=5 mm")
analyse(np.pi/1.0,"h=10 mm")
