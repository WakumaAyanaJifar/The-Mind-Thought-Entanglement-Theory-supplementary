# Paper: Section 3.5 and Supplementary S3, Fig. S3, Table S1. Expected: peak 0.954, spline FWHM 5.18, Hessian eigenvalues -1.230, 0, 0, 0.579, 0.651. Writes static2d.npy (needed by make_figs.py).
# Part of: the Mind-Thought Entanglement (MTE) framework - classical two-field model. Run via ../run_all.py
import numpy as np
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import LinearOperator, eigsh
mu,m,lam,g1,g2=1.0,1.0,1.0,0.5,1.98
N=128; L=40.0; x=(np.arange(N)-N//2)*L/N; X,Y=np.meshgrid(x,x)
k=2*np.pi*np.fft.fftfreq(N,L/N); KX,KY=np.meshgrid(k,k); K2=KX**2+KY**2
Kt=1/(K2+m*m)
conv=lambda f: np.real(np.fft.ifft2(np.fft.fft2(f)*Kt))
lap=lambda f: np.real(np.fft.ifft2(-K2*np.fft.fft2(f)))
def F(P):
    s=g1*P+0.5*g2*P**2
    return -lap(P)+mu**2*P+lam*P**3-(g1+g2*P)*conv(s)
res=None
for A in [1.0,1.5,2.0,2.5,3.0]:
    for w in [1.5,2.5,3.5]:
        P0=A*np.exp(-(X**2+Y**2)/w**2)
        try:
            P=newton_krylov(F,P0,f_tol=1e-10,maxiter=200,method='lgmres')
            if np.max(np.abs(P))>1e-3: res=P; print("converged from A,w",A,w); break
        except Exception as e: pass
    if res is not None: break
P=res
print("max residual",np.abs(F(P)).max(),"peak",P.max())
prof=P[N//2,:]; half=P.max()/2; above=x[prof>=half]
print("FWHM (grid points above half maximum):",above.max()-above.min()+L/N)
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
cs=CubicSpline(x,prof); i0=np.argmax(prof)
xr=brentq(lambda t:cs(t)-half,x[i0],x[i0]+L/4); xl=brentq(lambda t:cs(t)-half,x[i0]-L/4,x[i0])
print("FWHM (cubic-spline interpolation, value reported in the paper):",xr-xl)
# Hessian of static energy
def H(v):
    v=v.reshape(N,N); s=g1*P+0.5*g2*P**2
    out=-lap(v)+mu**2*v+3*lam*P**2*v-(g1+g2*P)*conv((g1+g2*P)*v)-g2*v*conv(s)
    return out.ravel()
op=LinearOperator((N*N,N*N),matvec=H,dtype=float)
ev=eigsh(op,k=5,which='SA',tol=1e-8,return_eigenvectors=False)
print("lowest Hessian eigenvalues:",np.sort(ev))
np.save('static2d.npy',P)
