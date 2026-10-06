# Paper: Figure 4. Uses recovery_sheet.py (model, fits) and recovery_sheet.npz (recovery map).
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as Pn
from scipy.special import j0, k0
import recovery_sheet as RS
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
 'axes.linewidth':0.7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
BL='#0072B2'; OR='#D55E00'; GR='#009E73'; GY='0.45'
RS.rng=np.random.default_rng(7)
def branches(th,k):
    c1,c2,g1,g2,a,m2,gg=th; out=[]
    for kk in k:
        L1=np.array([c1**2*kk**2+a,-1j*g1,-1.]); L2=np.array([c2**2*kk**2+m2,-1j*g2,-1.])
        w=Pn.polyroots(Pn.polysub(Pn.polymul(L1,L2),[gg])); w=np.sort(w[w.real>0].real); out.append(w)
    return np.array(out)
h=0.2; r,ru,inv=RS.grid(h); Ut=RS.field_transient(ru)[inv]
kfit=np.linspace(2*np.pi/12,min(np.pi/h,12.),40); J0k=j0(np.outer(kfit,r))*h**2
U=Ut+RS.rng.normal(0,np.sqrt(np.mean(Ut**2))/20,Ut.shape); F=J0k@U
th,_=RS.fit_P1(F,kfit)
fig,ax=plt.subplots(1,4,figsize=(7.2,2.15)); plt.subplots_adjust(left=0.06,right=0.99,bottom=0.21,top=0.87,wspace=0.45)
A=ax[0]; spec=np.abs(np.fft.rfft(F,axis=1)); fr=2*np.pi*np.fft.rfftfreq(len(RS.t),RS.DT)
sel=fr<8; Hz=fr[sel]/(2*np.pi*0.01)
A.pcolormesh(kfit,Hz,np.log10(spec[:,sel].T+1e-12),cmap='Greys',shading='auto',vmin=np.log10(spec.max())-3)
kk=np.linspace(kfit[0],kfit[-1],200); bt=branches(RS.th0,kk); bf=branches(th,kk)
for i in range(2): A.plot(kk,bt[:,i]/(2*np.pi*0.01),color=BL,lw=1.1); A.plot(kk,bf[:,i]/(2*np.pi*0.01),'--',color=OR,lw=1.0)
A.set_xlabel('wavenumber (rad cm$^{-1}$)'); A.set_ylabel('frequency (Hz)'); A.set_ylim(0,Hz.max())
A.set_title('a  transient spectrum',fontsize=9,loc='left')
# (b) sustained response, SNR 100, binned 1 mm
us0=(RS.kernel_profile(ru,RS.KAP0)@RS.B0)[inv]; us=us0+RS.rng.normal(0,np.sqrt(np.mean(us0**2))/100,us0.shape)
k4,R4=RS.fit_P4(us,lambda kap: RS.kernel_profile(ru,kap)[inv],RS.KAP0,RS.B0)
bins=np.arange(0,6.05,0.1); idx=np.digitize(r,bins); rb=[];ub=[]
for i in range(1,len(bins)):
    sel_=idx==i
    if sel_.any(): rb.append(r[sel_].mean()); ub.append(us[sel_].mean())
rb=np.array(rb); ub=np.array(ub)
rr=np.linspace(0.05,6,300); KP=lambda kap: RS.kernel_profile(rr,kap)
k1,B1,R1=RS.statics(th)
pred=KP(k1)@B1; fit=KP(k4)@np.linalg.lstsq(RS.kernel_profile(ru,k4)[inv],us,rcond=None)[0]
slow=KP(RS.KAP0[:1])[:,0]*RS.B0[0]
nrm=np.interp(1.0,rr,pred)
B=ax[1]; B.semilogy(rb*10,np.abs(ub)/nrm,'.',color=GY,ms=3,label='sensor data (1 mm bins)')
B.semilogy(rr*10,fit/nrm,color=BL,lw=1.1,label='two-component fit (P4)')
B.semilogy(rr*10,pred/nrm,'--',color=OR,lw=1.0,label='predicted from P1')
B.semilogy(rr*10,slow/nrm,':',color=GR,lw=1.1,label='slow component')
B.set_xlabel('distance (mm)'); B.set_ylabel('sustained response (norm.)'); B.set_ylim(1e-3,20); B.legend(fontsize=6,frameon=False,loc='upper right')
B.set_title('b  sustained response',fontsize=9,loc='left')
d=np.load('recovery_sheet.npz'); Hs=np.array(d['H']); S=np.array(d['SNR'])
C=ax[2]; im=C.imshow(d['frac'],origin='lower',cmap='viridis',vmin=0,vmax=1,aspect='auto')
for i in range(len(S)):
    for j in range(len(Hs)): C.text(j,i,f"{d['frac'][i,j]:.2f}",ha='center',va='center',fontsize=5.5,color='w' if d['frac'][i,j]<0.6 else 'k')
C.set_xticks(range(len(Hs))); C.set_xticklabels([f"{x*10:g}" for x in Hs]); C.set_yticks(range(len(S))); C.set_yticklabels(S)
C.set_xlabel('spacing (mm)'); C.set_ylabel('SNR'); C.set_title('c  joint recovery',fontsize=9,loc='left')
D=ax[3]
for j,(hh,col) in enumerate([(0.2,BL),(0.5,GR),(1.0,OR)]):
    jj=list(Hs).index(hh)
    D.loglog(S,np.maximum(d['E1'][:,jj],1e-4)*100,'o-',color=col,ms=3,lw=0.8,label=f'{hh*10:g} mm, P1')
    D.loglog(S,d['E4'][:,jj]*100,'s--',color=col,ms=3,mfc='none',lw=0.8,label=f'{hh*10:g} mm, P4')
D.set_xlabel('SNR'); D.set_ylabel(r'median error of $\kappa_{+}$ (%)'); D.legend(fontsize=5.5,frameon=False,ncol=2,loc='upper right')
D.set_ylim(0.1,4000); D.set_title(r'd  fast decay constant',fontsize=9,loc='left')
fig.savefig('fig4_recovery.pdf'); fig.savefig('fig4_recovery.png',dpi=150)
print('P1 fit kappa:',k1,'P4 fit kappa:',k4)
