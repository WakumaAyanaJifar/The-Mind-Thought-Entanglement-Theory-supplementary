"""Paper: Section 3.10 and Figure 4 (synthetic recovery of P1, P4 and the generalized identity).

Linear sector of the two-field model about the ordered state, damped and with unequal speeds
(parameters of Fig. 2b), on a homogeneous two-dimensional sheet. The field is driven at the
origin by (i) a brief pulse and (ii) a sustained drive, both of Gaussian spatial profile with
width sigma = 0.3. The linear equations are solved exactly (residues of the k-space impulse
response, inverse Hankel transform by Gauss-Legendre quadrature), sampled on a square grid of
spacing h over a patch of half-width 6, at time step 0.1 for t in (0, 30], and white noise is
added at a given signal-to-noise ratio (SNR = rms signal over sensors and samples / noise SD).

P1 route: isotropic spatial transform of the sensor data about the stimulation site,
          F(k,t) = h^2 sum_i u_i(t) J0(k r_i), fitted with A S(k) G(t,k; theta) for the seven
          dispersion parameters theta and the amplitude A; theta predicts kappa_+-, R_+-.
P4 route: the sustained profile fitted independently with two K0 components convolved with
          the known source, sum_j B_j (1/2pi) int k J0(kr) S(k)/(k^2+kappa_j^2) dk.
Each fit starts from parameters multiplied by independent factors U(0.6,1.6), bounded within a
factor 10 of the truth; failed fits count as failures. A realization is a recovery when both
routes give kappa_+ and kappa_- within 10% of the truth and of each other, and R_- within 0.1.

Usage: python recovery_sheet.py [--reps 40]   (writes recovery_sheet.npz and fig4 files)
"""
import numpy as np, sys, time
from numpy.polynomial import polynomial as Pn
from scipy.special import j0
from scipy.optimize import least_squares
rng = np.random.default_rng(20260930)
TRUE = dict(c1=1.0, c2=0.7, g1=0.6, g2=0.9, a=2.6875, m2=4.0, gg=6.25)
NAMES = ['c1','c2','g1','g2','a','m2','gg']
th0 = np.array([TRUE[n] for n in NAMES])
SIG, HALF, DT, TMAX = 0.3, 6.0, 0.1, 30.0
t = np.arange(DT, TMAX + DT/2, DT)
xq, wq = np.polynomial.legendre.leggauss(1400); KQ = 10.0*(xq+1); WQ = 10.0*wq   # k in [0,20]
S = lambda k: np.exp(-0.5*(k*SIG)**2)

def Gtk(th, k, tt):
    """Impulse response G(t,k) of the Psi field (unit impulse in the Psi equation)."""
    c1,c2,g1,g2,a,m2,gg = th
    out = np.empty((len(k), len(tt)))
    for i,kk in enumerate(k):
        L1 = np.array([c1**2*kk**2+a, -1j*g1, -1.0]); L2 = np.array([c2**2*kk**2+m2, -1j*g2, -1.0])
        D = Pn.polysub(Pn.polymul(L1,L2), [gg]); w = Pn.polyroots(D); dD = Pn.polyder(D)
        res = Pn.polyval(w, L2)/Pn.polyval(w, dD)
        out[i] = (-1j*(res[:,None]*np.exp(-1j*np.outer(w, tt)))).sum(0).real
    return out

def statics(th):
    c1,c2,g1,g2,a,m2,gg = th
    A2=c1**2*c2**2; A1=c1**2*m2+a*c2**2; A0=a*m2-gg
    q = np.sort(np.roots([A2,-A1,A0]).real); kap = np.sqrt(q)          # kappa_-, kappa_+
    B = np.array([(c2**2*(-x)+m2)/(2*A2*(-x)+A1) for x in q])          # chi(0,k)=sum B/(k^2+kap^2)
    return kap, B, B/B.sum()

def grid(h, half=HALF):
    n = int(round(half/h)); x = np.arange(-n, n+1)*h; X,Y = np.meshgrid(x,x)
    r = np.hypot(X,Y).ravel(); ru, inv = np.unique(np.round(r,9), return_inverse=True)
    return r, ru, inv

def field_transient(ru):
    kernel_profile(ru, [1.0]); G = Gtk(th0, KQ, t)
    return _JC[(len(ru), float(ru[-1]))] @ G                              # (n_r, n_t)

_JC = {}
def kernel_profile(ru, kap):
    key = (len(ru), float(ru[-1]))
    if key not in _JC: _JC.clear(); _JC[key] = j0(np.outer(ru, KQ))*(WQ*KQ*S(KQ))/(2*np.pi)
    J = _JC[key]
    return np.stack([J @ (1.0/(KQ**2+kp**2)) for kp in kap], 1)

def fit_P1(F, kfit):
    Sk = S(kfit)[:,None]; scale = np.abs(F).max()
    def model(p):
        th = np.exp(p[:7]); return np.exp(p[7])*Sk*Gtk(th, kfit, t)
    p_true = np.r_[np.log(th0), 0.0]
    p0 = p_true + np.log(rng.uniform(0.6,1.6,8))
    lb = p_true - np.log(10); ub = p_true + np.log(10)
    r = least_squares(lambda p: ((model(p)-F)/scale).ravel(), p0, bounds=(lb,ub), max_nfev=3000)
    return (np.exp(r.x[:7]) if r.success else None), r

def fit_P4(us, prof_fn, kap_true, B_true):
    scale = np.abs(us).max()
    def model(p):
        kap = np.exp(p[:2]); return prof_fn(kap) @ p[2:]
    p_true = np.r_[np.log(kap_true), B_true]
    p0 = np.r_[np.log(kap_true*rng.uniform(0.6,1.6,2)), B_true*rng.uniform(0.6,1.6,2)]
    lb = np.r_[np.log(kap_true/10), -10*np.abs(B_true)]; ub = np.r_[np.log(kap_true*10), 10*np.abs(B_true)]
    r = least_squares(lambda p: (model(p)-us)/scale, p0, bounds=(lb,ub), max_nfev=3000)
    if not r.success: return None
    kap = np.exp(r.x[:2]); B = r.x[2:]; o = np.argsort(kap)
    return kap[o], B[o]/B[o].sum()

KAP0, B0, R0 = statics(th0)

def run_condition(h, snr_list, reps, half=HALF):
    r, ru, inv = grid(h, half)
    Ut = field_transient(ru)[inv]                                       # (n_sensors, n_t)
    us0 = (kernel_profile(ru, KAP0) @ B0)[inv]                          # sustained, noise-free
    kfit = np.linspace(2*np.pi/(2*half), min(np.pi/h, 12.0), 40)
    J0k = j0(np.outer(kfit, r))*h**2
    prof_cache = {}
    def prof_fn(kap):
        return kernel_profile(ru, kap)[inv]
    rms_t = np.sqrt(np.mean(Ut**2)); rms_s = np.sqrt(np.mean(us0**2))
    out = {}
    for snr in snr_list:
        rec = []; e1 = []; e4 = []
        for rep in range(reps):
            if snr is None:
                U = Ut; us = us0
            else:
                U = Ut + rng.normal(0, rms_t/snr, Ut.shape); us = us0 + rng.normal(0, rms_s/snr, us0.shape)
            F = J0k @ U
            try: th, _ = fit_P1(F, kfit)
            except Exception: th = None
            try: p4 = fit_P4(us, prof_fn, KAP0, B0)
            except Exception: p4 = None
            if th is None or p4 is None:
                rec.append(False); e1.append(np.nan); e4.append(np.nan); continue
            k1, _, R1 = statics(th); k4, R4 = p4
            ok = (np.all(np.abs(k1/KAP0-1)<0.1) and np.all(np.abs(k4/KAP0-1)<0.1) and
                  np.all(np.abs(k1/k4-1)<0.1) and abs(R1[0]-R0[0])<0.1 and abs(R4[0]-R0[0])<0.1)
            rec.append(ok); e1.append(abs(k1[1]/KAP0[1]-1)); e4.append(abs(k4[1]/KAP0[1]-1))
        out[snr] = (np.mean(rec), np.nanmedian(e1), np.nanmedian(e4))
    return out, dict(r=r, Ut=Ut, us0=us0, kfit=kfit, J0k=J0k, ru=ru, inv=inv)

if __name__ == '__main__':
    reps = int(sys.argv[sys.argv.index('--reps')+1]) if '--reps' in sys.argv else 40
    H = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]; SNR = [5, 10, 20, 50, 100, 200, 500]
    print(f"truth: kappa_-={KAP0[0]:.4f} kappa_+={KAP0[1]:.4f} R_-={R0[0]:.4f}; 1/kappa_+={1/KAP0[1]:.4f}")
    frac = np.zeros((len(SNR), len(H))); E1 = np.zeros_like(frac); E4 = np.zeros_like(frac)
    for j,h in enumerate(H):
        t0 = time.time(); res, _ = run_condition(h, SNR, reps)
        for i,s in enumerate(SNR): frac[i,j], E1[i,j], E4[i,j] = res[s]
        print(f"h={h} ({h/(1/KAP0[1]):.2f}/kappa_+): recovered fraction by SNR {dict(zip(SNR, np.round(frac[:,j],3)))}"
              f"  median err kappa_+ P1 {np.round(E1[:,j]*100,2)}%  P4 {np.round(E4[:,j]*100,2)}%  [{time.time()-t0:.0f}s]", flush=True)
    # coverage check, noise-free, h=0.2: half-width 6 vs 4/kappa_-
    for half in [HALF, 4/KAP0[0]]:
        r, ru, inv = grid(0.2, half); Ut = field_transient(ru)[inv]
        kfit = np.linspace(2*np.pi/(2*half), min(np.pi/0.2, 12.0), 40)
        th, _ = fit_P1(j0(np.outer(kfit, r))*0.2**2 @ Ut, kfit); k1, _, R1 = statics(th)
        print(f"noise-free, half-width {half:.2f} ({half*KAP0[0]:.1f}/kappa_-): kappa bias {np.round((k1/KAP0-1)*100,2)}%, R_- {R1[0]:.4f}")
    np.savez('recovery_sheet.npz', H=H, SNR=SNR, frac=frac, E1=E1, E4=E4)
