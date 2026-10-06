"""Paper: Supplementary S3 and Table S1 (well-posedness). Adds a stabilizing sextic term
(nu/6) Psi^6 to the Psi potential and repeats the d=1 static computation and the +eps run:
the threshold state is essentially unchanged and the growing branch now saturates."""
import numpy as np
from scipy.optimize import newton_krylov
import static1d as S
mu,m,lam,g1,g2=S.mu,S.m,S.lam,S.g1,S.g2
def solve(nu,N=512,L=20.0):
    x,conv,lap=S.setup(N,L)
    F=lambda P: -lap(P)+mu**2*P+lam*P**3+nu*P**5-(g1+g2*P)*conv(g1*P+0.5*g2*P**2)
    P=newton_krylov(F,np.exp(-x**2/4),f_tol=1e-10,maxiter=300,method='lgmres'); return x,P,conv,lap
for nu in [0.0,0.01,0.05]:
    x,P,conv,lap=solve(nu); N=len(x); I=np.eye(N); H=np.empty((N,N)); s=g1*P+0.5*g2*P**2
    for j in range(N):
        v=I[j]; H[:,j]=-lap(v)+mu**2*v+3*lam*P**2*v+5*nu*P**4*v-(g1+g2*P)*conv((g1+g2*P)*v)-g2*v*conv(s)
    ev=np.linalg.eigvalsh(0.5*(H+H.T))[:3]
    # +eps dynamics along the unstable direction (gamma=0 replaced by weak damping 0.2 to settle)
    w,V=np.linalg.eigh(0.5*(H+H.T)); v0=V[:,0]*np.sign(V[np.argmax(P),0]); v0/=np.abs(v0).max()
    Ps=P+0.01*v0; ph=conv(g1*Ps+0.5*g2*Ps**2); vP=0*P; vph=0*P; dt=0.005; peak=[]
    for n in range(int(60/dt)):
        def f(Ps,ph,vP,vph):
            return vP,vph,lap(Ps)-mu**2*Ps-lam*Ps**3-nu*Ps**5+(g1+g2*Ps)*ph-0.2*vP, lap(ph)-m**2*ph+g1*Ps+0.5*g2*Ps**2-0.2*vph
        k1=f(Ps,ph,vP,vph); k2=f(*[a+dt/2*b for a,b in zip((Ps,ph,vP,vph),k1)])
        k3=f(*[a+dt/2*b for a,b in zip((Ps,ph,vP,vph),k2)]); k4=f(*[a+dt*b for a,b in zip((Ps,ph,vP,vph),k3)])
        Ps,ph,vP,vph=[a+dt/6*(p+2*q+2*r_+s_) for a,p,q,r_,s_ in zip((Ps,ph,vP,vph),k1,k2,k3,k4)]
        if not np.isfinite(Ps).all() or np.abs(Ps).max()>50: peak.append(np.inf); break
        if n%200==0: peak.append(Ps.max())
    end='diverges' if np.isinf(peak[-1]) else f"saturates at peak {peak[-1]:.2f}"
    print(f"nu={nu}: peak {P.max():.4f} FWHM {S.fwhm(x,P):.3f} lowest Hessian eigenvalues {np.round(ev,3)}; +eps run (weak damping 0.2): {end}")
