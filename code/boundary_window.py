"""Paper: Section 3.4 ('Which response diverges', 'Background drive', window of Eq. (window),
relation to steady-state TMS-EEG) and Supplementary S4. Parameters of Fig. 3.
(1) state-dependent chi(0), local response, xi, tau for wake, near-LOC and deep states;
(2) window budget; (3) bias scaling of LOC (linear) and ROC (square root);
(4) coexistence of P2 (sign channel) and P5 (measurable hysteresis)."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import exp1, k1
m=2.;lam=1.;g1=2.5;g2=1.2;c1=1.;c2=0.7;G1=.6;G2=.9; SIG=0.3
C=g1**2/m**2; b=g1*g2/(2*m**2); le=lam-g2**2/(2*m**2); W=9*b**2/(4*le)
def state(r,ordered):
    mu2=r+C
    f=lambda P: mu2*P+lam*P**3-(g1+g2*P)*(g1*P+g2*P**2/2)/m**2
    P=0.0
    if ordered:
        P=(3*b+2*np.sqrt(le*(W-r)))/(2*le)   # exact: the uniform equations reduce to V_eff'=0
    ph=(g1*P+g2*P**2/2)/m**2; a=mu2+3*lam*P**2-g2*ph; ge=g1+g2*P
    V=a-ge**2/m**2; c2e=(c1**2*m**2+a*c2**2)/m**2; G0=(a*G2+m**2*G1)/m**2
    z=SIG**2*V/(2*c2e); loc=np.exp(z)*exp1(z)/(4*np.pi*c2e)
    return dict(P=P,chi0=1/V,chi_loc=loc,xi=np.sqrt(c2e/V),tau=G0/V,ceff2=c2e,G0=G0)
print(f"Fig. 3 parameters: b={b}, lambda_eff={le}, W={W:.4f}")
print("(1) states (r, branch): chi(0), local response (sigma=0.3), xi, tau, ceff^2, damping coefficient")
for r,o,lab in [(-1.0,True,'wake'),(W-1e-3,True,'just before LOC'),(W,False,'just after LOC'),(1.0,False,'deep'),(4.0,False,'very deep')]:
    d=state(r,o); print(f"   r={r:+.3f} {lab:16s} chi0={d['chi0']:.3f} loc={d['chi_loc']:.4f} xi={d['xi']:.3f} tau={d['tau']:.3f} ceff2={d['ceff2']:.3f} G0={d['G0']:.3f}")
# (2) window budget
cL=state(W-1e-9,True)['ceff2']; cR=state(1e-9,False)['ceff2']
rG=1e-3*W; L=25.0; step=0.02*W
dL_LOC=64*cL**2/(W*L**4); dL_ROC=16*cR/L**2
Lam=np.pi/0.03; LG=np.log(cL*Lam**2/rG); D=4*np.pi*cL*rG/(3*le*LG)
I=62.006; Ln=np.log(100*6e4/(1.0*1.0))      # area 100, t_obs 6e4 time units, xi^2 tau ~ O(1)-O(100): conservative upper value
dstar=le*D*Ln/((I/2)*cL)
print(f"(2) window budget: r_G={rG:.2e} ({rG/W:.0e} W); D={D:.2e}; delta*(nucleation)<= {dstar:.2e} ({dstar/rG:.2f} r_G)")
print(f"    coverage L={L}: delta_L(LOC)={dL_LOC:.2e} ({dL_LOC/W:.4f} W), delta_L(ROC)={dL_ROC:.3f} ({dL_ROC/W:.3f} W); step={step:.4f}")
lo=max(rG,dstar,dL_LOC,step)/W
for x in [lo,10*lo]:
    print(f"    local LOC slope at delta/W={x:.3f}: {-(0.5*np.sqrt(x)+x)/(np.sqrt(x)+x):.3f}")
print(f"    ROC lower edge {max(rG,dL_ROC,step)/W:.3f} W; baseline drive must satisfy J0 << delta_min^2/(12 b) = {(dL_ROC)**2/(12*b):.1e}")
# (3) bias scaling
for J in [1e-4,4e-4,1.6e-3]:
    # analytic first order
    print(f"(3) J={J:.1e}: Delta r_LOC = 2 le J/(3b) = {2*le*J/(3*b):.2e}; ROC at sqrt(12bJ) = {np.sqrt(12*b*J):.4f}")
# numeric check: saddle-nodes are zeros of the discriminant of le P^3 - 3b P^2 + r P - J
def disc(r,J):
    A,B,Cc,Dd=le,-3*b,r,-J
    return 18*A*B*Cc*Dd-4*B**3*Dd+B**2*Cc**2-4*A*Cc**3-27*A**2*Dd**2
for J in [1e-4,4e-4,1.6e-3]:
    rROC=brentq(lambda r: disc(r,J),1e-6,0.2); rLOC=brentq(lambda r: disc(r,J),W-0.05,W+0.05)
    print(f"    numeric: ROC at r={rROC:.5f} (sqrt(12bJ)={np.sqrt(12*b*J):.5f}); LOC shift {rLOC-W:.3e} (J/Psi_c={2*le*J/(3*b):.3e})")
# (4) P2/P5 coexistence (effective potential, J=0, wake at r_w=-1)
print("(4) P2/P5 coexistence, lambda_eff=0.82, r_w=-1:")
for f in [0.01,0.02,0.05,0.1,0.2]:
    rw=-1.0; Wf=f/(1-f); bb=np.sqrt(4*0.82*Wf/9)
    s=np.sqrt(0.82*(Wf-rw)); Pp=(3*bb+2*s)/(2*0.82); Pm=(3*bb-2*s)/(2*0.82)
    V=lambda P: .5*rw*P**2-bb*P**3+0.82/4*P**4
    print(f"   W/(W-r_w)={f:.2f}: b={bb:.3f} Psi+={Pp:.3f} Psi-={Pm:.3f} barrier ratio {V(Pp)/V(Pm):.2f}")
