# Paper: Section 3.3 (damped example of Fig. 2b). Reproduces the k=0 resonances, kappa_-, kappa_+, R_- and both sides of identity (G1) = 3.030.
# Part of: the Mind-Thought Entanglement (MTE) framework - classical two-field model. Run via ../run_all.py
import numpy as np
# --- Reproduce paper's damped example (Fig 2b) ---
a=2.6875; m2=4.0; g2c=6.25; c1=1.0; c2=0.7; G1=0.6; G2=0.9
# quartic in w at k=0: (a - w^2 - i G1 w)(m2 - w^2 - i G2 w) - g^2
P1=np.poly1d([-1,-1j*G1,a]); P2=np.poly1d([-1,-1j*G2,m2])
Q=P1*P2-g2c
r=np.roots(Q.coeffs); print("k=0 roots",np.round(r,3))
# static poles: (a - c1^2 x)(m2 - c2^2 x) = g^2, x=kappa^2
S=np.poly1d([-c1**2,a])*np.poly1d([-c2**2,m2])-g2c
x=np.sort(np.roots(S.coeffs).real); km,kp=np.sqrt(x)
Rm=(m2/c2**2-km**2)/(kp**2-km**2)
print("kappa-,kappa+,R-",km,kp,Rm)
print("G1 lhs",km*kp," rhs",np.sqrt(np.prod(np.abs(r)))/(c1*c2))
