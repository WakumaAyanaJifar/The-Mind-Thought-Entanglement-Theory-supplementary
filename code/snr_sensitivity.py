"""Paper: Section 3.10 and Supplementary Table S6 (sensitivity of the required SNR).
Minimum SNR (from 50, 100, 200, 500, 1000) at which the joint recovery criterion of
recovery_sheet.py is met in >= 80% of 20 realizations, at a fixed spacing of 3 mm (h = 0.3),
when one parameter of Fig. 2b is multiplied by 0.8 or 1.25. Experimentalists can pass their own
parameter set: python snr_sensitivity.py c1 c2 gamma1 gamma2 a m2 geff2"""
import numpy as np, sys
import recovery_sheet as RS
base=RS.th0.copy(); names=['c1','c2','gamma1','gamma2','a','m2','geff2']
def required(th,reps=20,h=0.3,snrs=(50,100,200,500,1000)):
    c1,c2,g1,g2,a,m2,gg=th
    if a*m2-gg<=0: return 'unstable',None
    RS.th0=np.array(th); RS.KAP0,RS.B0,RS.R0=RS.statics(RS.th0); RS.rng=np.random.default_rng(5)
    res,_=RS.run_condition(h,list(snrs),reps)
    for s in snrs:
        if res[s][0]>=0.8: return s,RS.KAP0
    return '>1000',RS.KAP0
if len(sys.argv)==8:
    th=np.array([float(v) for v in sys.argv[1:]]); s,k=required(th); print(f"required SNR {s}; kappa={k}"); sys.exit()
s,k=required(base); print(f"baseline (Fig. 2b): required SNR {s}; kappa_-={k[0]:.3f} kappa_+={k[1]:.3f}",flush=True)
for i,nm in [(0,'c1'),(1,'c2'),(2,'gamma1'),(3,'gamma2'),(5,'m (m^2 x f^2)'),(6,'geff (geff^2 x f^2)')]:
    for f in [0.8,1.25]:
        th=base.copy(); th[i]*=f**2 if i in (5,6) else f
        s,k=required(th)
        ks='' if k is None else f"kappa_-={k[0]:.3f} kappa_+={k[1]:.3f}"
        print(f"{nm} x{f}: required SNR {s}  {ks}",flush=True)
