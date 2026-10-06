# Paper: Section 3.6 and Supplementary Fig. S4 / Table S1. Expected: Delta V = 0.2205 by direct minimization, against the first-order value 2 J Psi0 = 0.2250.
# Part of: the Mind-Thought Entanglement (MTE) framework - classical two-field model. Run via ../run_all.py
import numpy as np
from scipy.optimize import minimize_scalar
mu, C, lam, J = 1.0, 1.5625, 1.0, 0.15          # g2 = 0, so b = 0 and lambda_eff = lambda
V = lambda P: 0.5*(mu**2 - C)*P**2 + 0.25*lam*P**4 - J*P
Psi0 = np.sqrt((C - mu**2)/lam)
Pp = minimize_scalar(V, bounds=(0.0, 2.0), method='bounded', options={'xatol':1e-12}).x
Pm = minimize_scalar(V, bounds=(-2.0, 0.0), method='bounded', options={'xatol':1e-12}).x
print(f"Psi0 = {Psi0:.4f}; minima Psi+ = {Pp:.5f}, Psi- = {Pm:.5f}")
print(f"Delta V = V(Psi-) - V(Psi+) = {V(Pm)-V(Pp):.4f}   first order 2 J Psi0 = {2*J*Psi0:.4f}")
