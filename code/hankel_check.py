# Paper: Section 2.4 (numerical methods). Checks the closed two-component form of the sustained
# response, u(r) = sum_j B_j K0(kappa_j r)/(2 pi), by its numerical Hankel transform, which must
# return chi(0,k) = (c2^2 k^2 + m^2)/D(0,k), for the damped unequal-speed parameters of Fig. 2b and
# for the undamped equal-speed case.
import numpy as np
from scipy.special import k0, j0
from scipy.integrate import quad
def comps(c1,c2,a,m2,gg):
    A2=c1**2*c2**2; A1=c1**2*m2+a*c2**2; A0=a*m2-gg
    q=np.sort(np.roots([A2,-A1,A0]).real); B=np.array([(c2**2*(-x)+m2)/(2*A2*(-x)+A1) for x in q])
    return np.sqrt(q),B
for name,(c1,c2) in [('damped, unequal speeds',(1.0,0.7)),('undamped, equal speeds',(1.0,1.0))]:
    a,m2,gg=2.6875,4.0,6.25
    kap,B=comps(c1,c2,a,m2,gg)
    chi=lambda k:(c2**2*k**2+m2)/((c1**2*k**2+a)*(c2**2*k**2+m2)-gg)
    err=0
    for k in np.linspace(0.05,10,60):
        num=2*np.pi*sum(Bj/(2*np.pi)*quad(lambda r:r*j0(k*r)*k0(kj*r),0,60,limit=800,points=[1e-3,0.1,1])[0] for kj,Bj in zip(kap,B))
        err=max(err,abs(num/chi(k)-1))
    print(f"{name}: kappa={np.round(kap,4)}, max relative error over k in [0.05,10]: {err:.1e}")
