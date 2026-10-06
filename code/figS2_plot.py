# Paper: Supplementary Figure S2 (uses static1d.npz from static1d.py and envelope.npz from envelope.py).
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
 'axes.linewidth':0.7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
BL='#0072B2'; OR='#D55E00'; GR='#009E73'; GY='0.45'
s=np.load('static1d.npz'); e=np.load('envelope.npz')
fig,ax=plt.subplots(2,2,figsize=(7.0,4.6)); plt.subplots_adjust(left=0.08,right=0.98,bottom=0.1,top=0.94,wspace=0.28,hspace=0.42)
A=ax[0,0]; A.plot(s['x'],s['P'],color=BL,lw=1.4,label='exact equation (cross terms retained)')
A.plot(s['xn'],s['Pn'],'--',color=GY,lw=1.1,label='cross terms dropped')
A.set_xlim(-10,10); A.set_xlabel('$x$'); A.set_ylabel(r'$\Psi(x)$'); A.legend(fontsize=7,frameon=False,loc='upper left')
xs=np.linspace(-6,6,400); ins=A.inset_axes([0.7,0.55,0.27,0.4]); ins.plot(xs,np.exp(-np.abs(xs))/2,color=OR,lw=1); ins.set_title('kernel $K(x)$',fontsize=6.5); ins.tick_params(labelsize=6)
A.set_title('a  static localized solution ($d=1$)',fontsize=9,loc='left')
B=ax[0,1]; n=np.arange(1,7)
B.plot(n,s['ev'][:6],'o',color=BL,ms=5,label='exact'); B.plot(n+0.12,s['evn'][:6],'s',mfc='none',color=GY,ms=5,label='cross terms dropped')
B.axhline(0,color='k',lw=0.5); B.set_xlabel('eigenvalue index'); B.set_ylabel('Hessian eigenvalue'); B.legend(fontsize=7,frameon=False,loc='lower right')
B.annotate('unstable\ndirection',xy=(1,s['ev'][0]),xytext=(1.6,-1.2),fontsize=7,color=OR,arrowprops=dict(arrowstyle='-',color=OR,lw=0.6))
B.annotate('translation\nzero mode',xy=(2,0),xytext=(2.6,-0.75),fontsize=7,arrowprops=dict(arrowstyle='-',lw=0.6))
B.set_title('b  stability of the static solution',fontsize=9,loc='left')
C=ax[1,0]
for i,(lab,col) in enumerate([(r'$+\epsilon$ (grows)',OR),(r'$-\epsilon$ (decays)',BL),('unperturbed',GY)]):
    C.plot(s[f'dyn_t_{i}'],s[f'dyn_p_{i}'],color=col,lw=1.2,label=lab)
C.set_xlabel('time $t$'); C.set_ylabel(r'peak amplitude $\max\,\Psi$'); C.set_ylim(-0.5,6); C.legend(fontsize=7,frameon=False,loc='upper left')
C.set_title(r'c  full $\Psi$--$\phi_0$ dynamics from the static solution',fontsize=9,loc='left')
D=ax[1,1]
for A_,W_,lab,col,mk in [(e['A2'],e['W2'],'$d=2$',BL,'o'),(e['A3'],e['W3'],'$d=3$',GR,'s')]:
    ok=W_<60; D.loglog(A_[ok],W_[ok],mk,color=col,ms=3.5,label=lab)
    win=(W_>0.1)&(W_<10); p=np.polyfit(np.log(A_[win]),np.log(W_[win]),1)
    aa=np.logspace(np.log10(A_[win].min()),np.log10(A_[win].max()),10); D.loglog(aa,np.exp(np.polyval(p,np.log(aa))),'-',color=col,lw=0.8)
    print(f"slope {lab}: {p[0]:.3f}")
D.axhspan(0.1,10,color='0.92',lw=0,zorder=0); D.set_xlabel('peak amplitude $A$'); D.set_ylabel('width (FWHM)')
D.legend(fontsize=7,frameon=False,loc='upper right'); D.set_title(r'd  envelope states: $w\propto A^{-1/2}$ in the window',fontsize=9,loc='left')
fig.savefig('figS2_localized.pdf'); fig.savefig('figS2_localized.png',dpi=150)
