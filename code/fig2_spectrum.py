# Paper: Figure 2 (linear spectrum and sustained response). Parameters as in the caption.
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import k0
from scipy.optimize import minimize_scalar, brentq
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
 'axes.linewidth':0.7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
BL='#0072B2'; OR='#D55E00'; GR='#009E73'; GY='0.45'
mu,m,lam,g=1.0,2.0,1.0,2.5
fig,ax=plt.subplots(1,4,figsize=(7.2,2.05)); plt.subplots_adjust(left=0.06,right=0.99,bottom=0.2,top=0.88,wspace=0.42)
# (a) symmetric state, undamped, equal speeds
k=np.linspace(0,3,400); a0=mu**2
Xp=0.5*(a0+m**2+np.sqrt((a0-m**2)**2+4*g**2)); Xm=0.5*(a0+m**2-np.sqrt((a0-m**2)**2+4*g**2))
kc=np.sqrt(-Xm)
A=ax[0]; A.axvspan(0,kc,color='0.9',lw=0)
A.plot(k,k**2+Xp,color=BL,lw=1.2); A.plot(k,k**2+Xm,color=OR,lw=1.2)
A.plot(k,k**2+mu**2,'--',color=GY,lw=0.8); A.plot(k,k**2+m**2,'--',color=GY,lw=0.8)
A.axhline(0,color='k',lw=0.5); A.set_xlabel('wavenumber $k$'); A.set_ylabel(r'$\omega^{2}$')
A.text(kc+0.05,-1.5,r'$k_{c}\approx%.3f$'%kc,fontsize=7.5); A.set_xlim(0,3); A.set_ylim(-2,15)
A.set_title('a  symmetric state',fontsize=9,loc='left')
# (b) ordered state, damped, unequal speeds: spectral density of Psi response
P0=np.sqrt((g**2/m**2-mu**2)/lam); a=mu**2+3*lam*P0**2; c1,c2,G1,G2=1.0,0.7,0.6,0.9
kk=np.linspace(0.02,6,300); ww=np.linspace(0.02,7,300); K,Wg=np.meshgrid(kk,ww)
num=c2**2*K**2+m**2-Wg**2-1j*G2*Wg
chi=num/((c1**2*K**2+a-Wg**2-1j*G1*Wg)*num-g**2)
B=ax[1]; B.pcolormesh(K,Wg,np.log10(np.abs(chi.imag)+1e-4),cmap='Greys',shading='auto',vmin=-2.5,vmax=1)
Xp2=0.5*(a+m**2+np.sqrt((a-m**2)**2+4*g**2)); Xm2=0.5*(a+m**2-np.sqrt((a-m**2)**2+4*g**2))
B.plot(kk,np.sqrt(kk**2+Xp2),'--',color=OR,lw=0.9); B.plot(kk,np.sqrt(kk**2+Xm2),'--',color=BL,lw=0.9)
B.plot(kk,c1*kk,':',color='k',lw=0.8); B.plot(kk,c2*kk,':',color='k',lw=0.8)
B.set_xlim(0,6); B.set_ylim(0,7); B.set_xlabel('wavenumber $k$'); B.set_ylabel(r'frequency $\omega$')
B.set_title('b  ordered, damped',fontsize=9,loc='left')
# static two-component profiles: chi(q)=(c2^2 q+m^2)/D(q); decay constants and weights
def comps(c1,c2,a,m2,gg):
    A2=c1**2*c2**2; A1=c1**2*m2+a*c2**2; A0=a*m2-gg
    q=np.roots([A2,-A1,A0])   # in kappa^2: D(-kappa^2)=A2 kappa^4 - A1 kappa^2 + A0
    kap=np.sqrt(np.sort(q.real)); w=[]
    for kp in kap:
        x=-kp**2; dD=2*A2*x+A1
        w.append((c2**2*x+m2)/dD)
    w=np.array(w); return kap,w/w.sum()
kapU,RU=comps(1,1,a,m**2,g**2)            # undamped equal speed
kapD,RD=comps(c1,c2,a,m**2,g**2)          # damped unequal speed (static part)
r=np.linspace(0.05,6,600)
prof=lambda kap,R: R[0]*k0(kap[0]*r)+R[1]*k0(kap[1]*r)
uU=prof(kapU,RU); n1=np.interp(1,r,uU)
# best single K0 fit in log over r in [0.2,5]
sel=(r>0.2)&(r<5)
def cost(kp):
    s=k0(kp*r[sel]); A_=np.exp(np.mean(np.log(uU[sel])-np.log(s))); return np.mean((np.log(uU[sel])-np.log(A_*s))**2)
kp1=minimize_scalar(cost,bounds=(0.3,3),method='bounded').x
u1=k0(kp1*r); u1*=n1/np.interp(1,r,u1)
# E-I example: opposite-sign cross coupling (alpha12*alpha21=-geI2), real decay constants
aE,mE,geI=4.0,1.0,1.45  # m_E^2=1 < kappa_-^2 puts the negative weight on the slow component
kapE,RE=comps(1,1,aE,mE,-geI**2)
uE=prof(kapE,RE)
C=ax[2]; C.axhline(0,color='k',lw=0.5)
C.plot(r,uU/n1,color=BL,lw=1.3,label='two-field model')
C.plot(r,u1/n1,'--',color=GY,lw=1.0,label=r'single $K_0$ (best fit)')
C.plot(r,uE/np.interp(1,r,uE),color=OR,lw=1.1,label='opposite-sign coupling')
C.set_xlim(0,6); C.set_ylim(-0.5,3.2); C.set_xlabel('distance $r$'); C.set_ylabel('response (norm. at $r=1$)')
C.legend(fontsize=6.5,frameon=False,loc='upper right'); C.set_title('c  sustained response, $d=2$',fontsize=9,loc='left')
# (d) effective decay constant: sum R K0(kap r) = K0(keff r)
rd=np.logspace(np.log10(0.02),np.log10(60),300)
from scipy.special import k0e
def keff(kap,R):
    out=[]
    for rr in rd:
        lv=np.log(R[0]*k0e(kap[0]*rr)*np.exp(-(kap[0]-kap[0])*rr)+R[1]*k0e(kap[1]*rr)*np.exp(-(kap[1]-kap[0])*rr))-kap[0]*rr
        f=lambda x: np.log(k0e(x*rr))-x*rr-lv
        out.append(brentq(f,1e-6,60/rr+60))
    return np.array(out)
D_=ax[3]
D_.semilogx(rd,keff(kapU,RU),color=BL,lw=1.3,label='undamped, equal speeds')
D_.semilogx(rd,keff(kapD,RD),color=GR,lw=1.3,label='damped, unequal speeds')
D_.semilogx(rd,np.full_like(rd,kp1),'--',color=GY,lw=1.0,label=r'single $K_0$')
for kp in [kapU[0],kapD[0]]: D_.axhline(kp,ls=':',color='k',lw=0.6)
D_.set_xlabel('distance $r$'); D_.set_ylabel(r'$\kappa_{\mathrm{eff}}(r)$'); D_.legend(fontsize=6.5,frameon=False,loc='upper right')
D_.set_ylim(0.6,2.1); D_.set_title(r'd  effective decay constant',fontsize=9,loc='left')
fig.savefig('fig2.pdf'); fig.savefig('fig2.png',dpi=150)
gmU=np.exp(RU@np.log(kapU)); gmD=np.exp(RD@np.log(kapD))
print(f"k_c={kc:.4f}; Psi0={P0}; a={a}")
print(f"undamped equal-speed: kappa={kapU}, R={RU}, geometric mean={gmU:.3f}")
print(f"damped unequal-speed: kappa={kapD}, R={RD}, geometric mean={gmD:.3f}")
print(f"single K0 best fit kappa={kp1:.3f}; E-I example kappa={kapE}, weights={RE}")
