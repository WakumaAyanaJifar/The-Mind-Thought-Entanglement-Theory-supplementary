# Paper: Section 3.4 (nucleation barrier 31.0 c^2 delta/lambda_eff; local log law, Eq. (21)) and Supplementary S4.
"""2D critical nucleus near the LOC spinodal and the local (log) amplitude law.
(1) Bounce of -lap u + u - u^2 = 0 in d=2 by shooting; nucleus barrier
    F = I c^2 delta/(2 lambda_eff), I = 8 E_u, E_u = int[ (1/2)|grad u|^2 + u^2/2 - u^3/3 ].
(2) Local response to a Gaussian source of width sigma: exact
    chi_loc = exp(z) E1(z)/(4 pi c^2), z = sigma^2 V''/(2 c^2), checked by quadrature."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy.special import exp1
def shoot(u0,R=30):
    f=lambda r,y:[y[1],-y[1]/r+y[0]-y[0]**2]
    r0=1e-6; y0=[u0+(u0-u0**2)*r0**2/4,(u0-u0**2)*r0/2]
    ev=lambda r,y: y[0]; ev.terminal=True
    ev2=lambda r,y: y[1]; ev2.terminal=True
    s=solve_ivp(f,[r0,R],y0,events=[ev,ev2],rtol=1e-12,atol=1e-14,dense_output=True)
    return s
def miss(u0):
    s=shoot(u0); return -1 if len(s.t_events[0]) else 1   # crosses zero -> overshoot
lo,hi=1.5,3.5
for _ in range(60):
    mid=(lo+hi)/2
    if miss(mid)<0: hi=mid
    else: lo=mid
u0=(lo+hi)/2
s=shoot(u0,R=12); r=np.linspace(1e-6,min(s.t[-1],12),20001); u,up=s.sol(r)
E=np.trapezoid(2*np.pi*r*(0.5*up**2+0.5*u**2-u**3/3),r)
Egrad=np.trapezoid(2*np.pi*r*0.5*up**2,r)
print(f"bounce u(0) = {u0:.5f}; E_u = {E:.4f} (virial: gradient part {Egrad:.4f}); I = 8 E_u = {8*E:.3f}")
# local amplitude check
c2=1.3;sig=0.3
for Vpp in [1e-1,1e-2,1e-3]:
    num=quad(lambda k: k*np.exp(-k**2*sig**2/2)/(Vpp+c2*k**2),0,np.inf,limit=400)[0]/(2*np.pi)
    z=sig**2*Vpp/(2*c2); ex=np.exp(z)*exp1(z)/(4*np.pi*c2)
    print(f"V''={Vpp:g}: quadrature {num:.6f}  exact {ex:.6f}  log form {(np.log(2*c2/(sig**2*Vpp))-np.euler_gamma)/(4*np.pi*c2):.6f}")
