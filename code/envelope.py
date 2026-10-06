"""Paper: Supplementary S3, Fig. S2d (width-amplitude scaling of envelope ground states).
Fixed-norm ground states of  -(1/2mu) lap psi + alpha |psi|^2 psi - beta (K*|psi|^2) psi = E psi,
K the Yukawa kernel (-lap+m^2)^-1, in d=2 and d=3; radial grid of 2400 logarithmically spaced
points on [1e-5, 4000]; exact radial Green's functions; self-consistent-field iteration.
alpha = 3 lambda/(8 mu^2) = 1e-4, beta = g2^2/(8 mu^2) = 1, m = 1e-2, mu = 1."""
import numpy as np
from scipy.special import k0e, i0e
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import cumulative_trapezoid as ctr
mu,alpha,beta,m=1.0,1e-4,1.0,1e-2
r=np.logspace(-5,np.log10(4000),2400)
def yukawa(rho,d):
    if d==3:
        mr=m*r; a=np.exp(-mr)*ctr(np.sinh(m*r)*rho*r,r,initial=0)
        b=np.sinh(mr)*(ctr((np.exp(-m*r)*rho*r)[::-1],r[::-1],initial=0)[::-1]*-1)
        return (a+b)/(m*r)
    else:
        # K0(mr) int_0^r I0 rho r' dr' + I0(mr) int_r^inf K0 rho r' dr'   (scaled Bessels for stability)
        A=ctr(i0e(m*r)*np.exp(m*r)*rho*r,r,initial=0)
        B=-ctr((k0e(m*r)*np.exp(-m*r)*rho*r)[::-1],r[::-1],initial=0)[::-1]
        return k0e(m*r)*np.exp(-m*r)*A+i0e(m*r)*np.exp(m*r)*B
def ground(Nn,d,psi=None,it=400):
    # finite-volume radial operator with weights w=r^(d-1); symmetric form via u=sqrt(w*dr) psi
    rh=np.sqrt(r[1:]*r[:-1]); wf=rh**(d-1)/(r[1:]-r[:-1])        # face conductances
    dr=np.gradient(r); vol=r**(d-1)*dr
    if psi is None: psi=np.exp(-r**2/(2*(Nn**(-1.0/(4-d))+1e-3)**2))
    else: psi=psi*np.sqrt(Nn/np.sum(psi**2*r**(d-1)*np.gradient(r)))
    for k in range(it):
        rho=psi**2; V=alpha*rho-beta*yukawa(rho,d)
        diag=np.zeros_like(r); diag[:-1]+=wf; diag[1:]+=wf
        diag=diag/(2*mu)/vol+V; off=-(wf/(2*mu))/np.sqrt(vol[1:]*vol[:-1])
        E,u=eigh_tridiagonal(diag,off,select='i',select_range=(0,0))
        new=np.abs(u[:,0])/np.sqrt(vol); new*=np.sqrt(Nn/np.sum(new**2*vol))
        if np.max(np.abs(new-psi))<1e-9*new.max(): psi=new; break
        psi=0.5*psi+0.5*new
    return psi
def width(psi):
    A=psi[0]; i=np.where(psi<A/2)[0][0]
    rr=np.interp(np.log(A/2),np.log(psi[i-1:i+1][::-1]),r[i-1:i+1][::-1])
    return 2*rr,A
if __name__=='__main__':
    res={}
    for d in [2,3]:
        Ns=np.logspace(-0.5,3.5,33) if d==3 else np.logspace(-1.5,5.5,36)
        W=[];A=[];psi=None
        for Nn in Ns[::-1]:
            psi=ground(Nn,d,psi,it=3000)
            w,a=width(psi); W.append(w); A.append(a)
        W=np.array(W[::-1]);A=np.array(A[::-1]); res[d]=(A,W)
        win=(W>0.1)&(W<10)   # well inside sqrt(alpha/beta)=0.01 << w << 1/m=100
        sl=np.polyfit(np.log(A[win]),np.log(W[win]),1)[0]
        print("  widths:",np.round(W,4)); print(f"d={d}: fitted log-log slope of width vs amplitude in the window = {sl:.3f} ({win.sum()} states)")
    np.savez('envelope.npz',A2=res[2][0],W2=res[2][1],A3=res[3][0],W3=res[3][1])
