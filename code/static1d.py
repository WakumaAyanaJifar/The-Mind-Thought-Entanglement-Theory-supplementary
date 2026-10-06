"""Paper: Supplementary S3, Fig. S2a-c and Table S1 (d=1 static threshold state).
Static equation with cross terms retained, and with the g1*g2 cross terms dropped;
initial-guess survey; grid convergence; g2 sensitivity; Hessian; full Psi-phi0 dynamics."""
import numpy as np
from scipy.optimize import newton_krylov
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
mu,m,lam,g1,g2=1.0,1.0,1.0,0.5,1.98
def setup(N,L):
    x=(np.arange(N)-N//2)*L/N; k=2*np.pi*np.fft.fftfreq(N,L/N)
    Kt=1/(k**2+m*m); conv=lambda f: np.real(np.fft.ifft(np.fft.fft(f)*Kt))
    lap=lambda f: np.real(np.fft.ifft(-k**2*np.fft.fft(f))); return x,conv,lap
def Ffun(conv,lap,g2v,cross=True):
    def F(P):
        if cross: rhs=(g1+g2v*P)*conv(g1*P+0.5*g2v*P**2)
        else: rhs=g1**2*conv(P)+0.5*g2v**2*P*conv(P**2)
        return -lap(P)+mu**2*P+lam*P**3-rhs
    return F
def solve(N=512,L=20.0,g2v=g2,cross=True,A=1.0,w=2.0):
    x,conv,lap=setup(N,L); F=Ffun(conv,lap,g2v,cross)
    P=newton_krylov(F,A*np.exp(-x**2/w**2),f_tol=1e-10*max(1,N/512),maxiter=300,method='lgmres')
    return x,P,np.abs(F(P)).max()
def fwhm(x,P):
    cs=CubicSpline(x,P); i=np.argmax(P); h=P[i]/2
    return brentq(lambda t:cs(t)-h,x[i],x[-1])-brentq(lambda t:cs(t)-h,x[0],x[i])
def hessian(x,P,g2v=g2,cross=True):
    N=len(x); L=(x[1]-x[0])*N; _,conv,lap=setup(N,L); I=np.eye(N); H=np.empty((N,N))
    s=g1*P+0.5*g2v*P**2
    for j in range(N):
        v=I[j]
        if cross: H[:,j]=-lap(v)+mu**2*v+3*lam*P**2*v-(g1+g2v*P)*conv((g1+g2v*P)*v)-g2v*v*conv(s)
        else: H[:,j]=-lap(v)+mu**2*v+3*lam*P**2*v-g1**2*conv(v)-0.5*g2v**2*v*conv(P**2)-g2v**2*P*conv(P*v)
    H=0.5*(H+H.T); return np.linalg.eigh(H)
if __name__=='__main__':
    x,P,res=solve(); print(f"full: peak {P.max():.4f} FWHM {fwhm(x,P):.3f} residual {res:.1e}")
    xn,Pn_,resn=solve(cross=False,A=1.3,w=2.0); print(f"no cross terms: peak {Pn_.max():.4f} FWHM {fwhm(xn,Pn_):.3f}")
    _,conv,_=setup(512,20.0); i=np.argmax(P)
    hart=0.5*g2**2*conv(P**2)[i]*P[i]; cross=g1*g2*(0.5*conv(P**2)[i]+P[i]*conv(P)[i])
    print(f"cross/Hartree at peak: {cross/hart:.2f}")
    ev,V=hessian(x,P); print("Hessian (full) lowest:",np.round(ev[:3],3))
    evn,_=hessian(xn,Pn_,cross=False); print("Hessian (no cross) lowest:",np.round(evn[:3],3))
    nsol=ntriv=nother=0
    for A in np.linspace(0.3,2.7,9):
        for w in np.linspace(0.5,4.5,9):
            try:
                _,Q,_=solve(A=A,w=w)
                if abs(Q.max()-P.max())<1e-4: nsol+=1
                elif np.abs(Q).max()<1e-6: ntriv+=1
                else: nother+=1
            except Exception: nother+=1
    print(f"initial-guess survey (9x9): {nsol} to the solution, {ntriv} to the trivial state, {nother} other/failed")
    for N,L in [(256,20),(1024,20),(2048,20),(512,40),(1024,40)]:
        xx,Q,_=solve(N=N,L=float(L)); print(f"  N={N} L={L}: peak {Q.max():.5f} FWHM {fwhm(xx,Q):.4f}")
    for f in [0.95,1.05]:
        xx,Q,_=solve(g2v=g2*f); print(f"  g2 x{f}: peak {Q.max():.3f} FWHM {fwhm(xx,Q):.2f}")
    # dynamics (gamma=0): unstable eigenvector, phi displaced consistently
    v0=V[:,0]*np.sign(V[np.argmax(P),0]); v0/=np.abs(v0).max()
    _,conv,lap=setup(512,20.0); dt=0.005; T=30.0
    def rhs(Y):
        Ps,ph,vP,vph=Y
        return np.array([vP,vph,lap(Ps)-mu**2*Ps-lam*Ps**3+(g1+g2*Ps)*ph, lap(ph)-m**2*ph+g1*Ps+0.5*g2*Ps**2])
    out={}
    for eps in [0.01,-0.01,0.0]:
        Ps=P+eps*v0; ph=conv(g1*Ps+0.5*g2*Ps**2); Y=np.array([Ps,ph,0*P,0*P]); ts=[0.0]; pk=[Ps.max()]
        n=0
        while ts[-1]<T and np.abs(Y[0]).max()<6:
            k1=rhs(Y);k2=rhs(Y+dt/2*k1);k3=rhs(Y+dt/2*k2);k4=rhs(Y+dt*k3); Y=Y+dt/6*(k1+2*k2+2*k3+k4); n+=1
            if n%20==0: ts.append(n*dt); pk.append(Y[0].max())
        out[eps]=(np.array(ts),np.array(pk)); print(f"  dynamics eps={eps:+.2f}: final t={ts[-1]:.2f}, peak={pk[-1]:.3g}")
    np.savez('static1d.npz',x=x,P=P,xn=xn,Pn=Pn_,ev=ev[:8],evn=evn[:8],
             **{f'dyn_t_{i}':out[e][0] for i,e in enumerate([0.01,-0.01,0.0])},
             **{f'dyn_p_{i}':out[e][1] for i,e in enumerate([0.01,-0.01,0.0])})
