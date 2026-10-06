# Paper: generates Fig. 1 (fig1_rev), Fig. 3 (fig3_boundary) and Fig. S3 (figS3_static2d). Requires static2d.npy from static2d.py in the working directory.
# Part of: the Mind-Thought Entanglement (MTE) framework - classical two-field model. Run via ../run_all.py
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
                     'axes.linewidth':0.7,'xtick.direction':'in','ytick.direction':'in',
                     'xtick.top':True,'ytick.right':True})
BL='#0072B2'; OR='#D55E00'; GR='#009E73'; GY='0.45'
# ---------------- Figure 1 (revised): core + linked test ----------------
fig=plt.figure(figsize=(7.0,2.9))
ax=fig.add_axes([0.0,0.0,0.46,1.0]); ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis('off')
ax.text(0.2,6.6,'a',fontsize=12,fontweight='bold')
def box(x,y,w,h,fc,ec,txt,fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.3',fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,linespacing=1.35)
box(0.4,3.7,3.6,2.5,'#E8F1FA',BL,'$\\Psi(\\mathbf{x},t)$\nglobal order\nparameter (observed)\n$\\partial_t^2+\\gamma_1\\partial_t-c_1^2\\nabla^2+\\mu^2$',8.5)
box(5.9,3.7,3.6,2.5,'#FCEBDD',OR,'$\\phi_0(\\mathbf{x},t)$\nhidden context\nmode\n$\\partial_t^2+\\gamma_2\\partial_t-c_2^2\\nabla^2+m^2$',8.5)
ax.add_patch(FancyArrowPatch((4.15,5.2),(5.75,5.2),arrowstyle='<|-|>',mutation_scale=10,lw=1.1,color='k'))
ax.text(4.95,5.55,'$g_1\\Psi+\\frac{g_2}{2}\\Psi^2$',ha='center',fontsize=8.5)
ax.text(4.95,4.55,'same terms\nboth ways',ha='center',fontsize=7.5,color=GY)
ax.add_patch(FancyArrowPatch((2.2,3.55),(2.2,2.55),arrowstyle='-|>',mutation_scale=10,lw=1.0,color=GY))
ax.text(2.45,3.0,'eliminate $\\phi_0$ exactly',fontsize=8,color=GY,va='center')
box(0.4,0.3,9.1,2.0,'white','0.35','one nonlocal equation for $\\Psi$ with retarded kernel\n'
    '$\\widetilde{K}(\\omega,\\mathbf{k})=1/(c_2^2k^2+m^2-\\omega^2-i\\gamma_2\\omega)$;  range $c_2/m$, memory $2/\\gamma_2$',8.5)
# panel b: linked test
ax2=fig.add_axes([0.56,0.14,0.42,0.76])
Xm,Xp=0.7591,5.9284
k2=np.linspace(0,5.5,50); k2n=np.linspace(-7.5,0,50)
ax2.axvspan(-7.5,0,color='0.93'); ax2.axhline(0,color='k',lw=0.6); ax2.axvline(0,color='k',lw=0.6)
for X,c,l in [(Xp,OR,'$\\omega_+^2$'),(Xm,BL,'$\\omega_-^2$')]:
    ax2.plot(k2,k2+X,c=c,lw=1.6); ax2.plot(k2n,k2n+X,c=c,lw=1.2,ls='--')
    ax2.plot(-X,0,'o',mfc='white',mec=c,mew=1.4,ms=6)
    ax2.text(3.0,3.0+X+0.6,l,color=c,fontsize=9)
ax2.text(-X-0.3,-1.2,'$-\\kappa_+^2$' if X==Xp else '',color=OR)
ax2.text(-Xp-0.5,-1.3,'$-\\kappa_+^2$',color=OR,fontsize=8.5); ax2.text(-Xm-0.4,-1.3,'$-\\kappa_-^2$',color=BL,fontsize=8.5)
ax2.text(-7.2,10.2,'sustained response (P4)\n$\\omega=0,\\ k=i\\kappa$',fontsize=8)
ax2.text(0.4,-1.6,'transient spectrum (P1)',fontsize=8)
ax2.set_xlim(-7.5,5.5); ax2.set_ylim(-2.2,12.5); ax2.set_xlabel('$k^2$'); ax2.set_ylabel('$\\omega^2$')
ax2.set_title('decay constants = zero-frequency roots',fontsize=9)
fig.text(0.53,0.93,'b',fontsize=12,fontweight='bold')
fig.savefig('fig1_rev.pdf'); fig.savefig('fig1_rev.png',dpi=130); plt.close(fig)

# ---------------- Figure 3 (new): asymmetric boundary ----------------
m=2.0; lam=1.0; g1=2.5; g2=1.2; c1=1.0; c2=0.7; G1=0.6; G2=0.9
m2=m*m; C=g1**2/m2; b=g1*g2/(2*m2); leff=lam-g2**2/(2*m2); W=9*b**2/(4*leff)
def roots(r,J=0.0):
    rt=np.roots([leff,-3*b,r,-J]); return np.sort(rt[np.abs(rt.imag)<1e-10].real)
def lin(P,r):
    mu2=r+C; phi=(g1*P+g2*P**2/2)/m2; a=mu2+3*lam*P**2-g2*phi; ge=g1+g2*P
    q=np.polymul([-1,-1j*G1,a],[-1,-1j*G2,m2]); q[-1]-=ge**2; w=np.roots(q)
    s=np.polymul([-c1**2,a],[-c2**2,m2]); s[-1]-=ge**2; x=np.sort(np.roots(s).real)
    return 1/(a-ge**2/m2), 1/np.min(-w.imag), 1/np.sqrt(x[0])
fig,axs=plt.subplots(1,4,figsize=(7.2,2.35)); plt.subplots_adjust(left=0.07,right=0.99,bottom=0.2,top=0.86,wspace=0.45)
# (a) order parameter, dose axis = r/W
r_up=np.linspace(-0.6*W,1.6*W,600)
P_up=np.array([roots(r)[-1] if r<W else 0.0 for r in r_up])
P_dn=np.array([0.0 if r>0 else roots(r)[-1] for r in r_up])
ax=axs[0]
ax.axvspan(0,1,color='#FCEBDD',lw=0)
ax.plot(r_up/W,P_up,c=OR,lw=1.5,label='induction')
ax.plot(r_up/W,P_dn,c=BL,lw=1.5,ls='--',label='emergence')
Pb0=np.sqrt(np.clip(-(r_up)/lam,0,None)); ax.plot(r_up/W,Pb0,c=GY,lw=0.9,ls=':',label='$b=0$')
ax.annotate('',xy=(1,0.03),xytext=(1,roots(W*0.999)[-1]),arrowprops=dict(arrowstyle='->',color=OR,lw=1))
ax.annotate('',xy=(0,roots(-1e-9)[-1]-0.03),xytext=(0,0.03),arrowprops=dict(arrowstyle='->',color=BL,lw=1))
ax.text(1.03,0.55,'LOC',color=OR,fontsize=8); ax.text(-0.5,0.2,'ROC',color=BL,fontsize=8)
ax.set_xlabel('dose, $(\\mu^2-C)/W$'); ax.set_ylabel('$\\Psi_0$'); ax.set_title('a  hysteresis',loc='left',fontsize=9)
ax.legend(fontsize=6.5,frameon=False,loc='upper right',handlelength=1.6)
# (b) susceptibility along the two sweeps
chi_up=[]; chi_dn=[]
for r,pu,pd in zip(r_up,P_up,P_dn):
    chi_up.append(lin(pu,r)[0] if not (pu==0 and abs(r)<1e-12) else np.nan)
    chi_dn.append(lin(pd,r)[0] if abs(r)>1e-9 else np.nan)
chi_up=np.array(chi_up); chi_dn=np.array(chi_dn)
chi_dn[(r_up>0)&(r_up<0.004*W)]=np.nan
ax=axs[1]; ax.axvspan(0,1,color='#FCEBDD',lw=0)
ax.semilogy(r_up/W,chi_up,c=OR,lw=1.5); ax.semilogy(r_up/W,chi_dn,c=BL,lw=1.5,ls='--')
ax.set_ylim(0.3,300); ax.set_xlabel('dose, $(\\mu^2-C)/W$'); ax.set_ylabel('$\\widetilde\\chi(0)$')
ax.set_title('b  two separate peaks',loc='left',fontsize=9)
ax.text(1.05,40,'LOC',color=OR,fontsize=8); ax.text(0.08,120,'ROC',color=BL,fontsize=8)
# (c) exponents
d=np.logspace(-5,-1.3,30)*W
LOC=np.array([lin(roots(W-x)[-1],W-x) for x in d]); ROC=np.array([lin(0.0,x) for x in d])
ax=axs[2]
for i,(lab,mk) in enumerate([('$\\widetilde\\chi$','-'),('$\\tau$','-.'),('$\\xi$',':')]):
    ax.loglog(d/W,LOC[:,i]/LOC[-1,i],c=OR,ls=mk,lw=1.3)
    ax.loglog(d/W,ROC[:,i]/ROC[-1,i],c=BL,ls=mk,lw=1.3,label=lab)
ax.text(2.5e-4,1.5e3,'ROC: 1, 1, ½',color=BL,fontsize=7.5); ax.text(2e-5,1.2,'LOC: ½, ½, ¼',color=OR,fontsize=7.5)
ax.set_xlabel('distance from transition, $\\delta/W$'); ax.set_ylabel('normalized'); ax.legend(fontsize=6.5,frameon=False,loc='center right')
ax.set_title('c  exponents',loc='left',fontsize=9)
# (d) parameter-free ratio
dd=np.logspace(-3,0,40)*W
num=np.array([lin(roots(W-x)[-1],W-x)[0]/lin(0.0,x)[0] for x in dd])
ax=axs[3]; ax.semilogx(dd/W,1/(2*(1+np.sqrt(W/dd))),c='k',lw=1.2,label='closed form')
ax.semilogx(dd[::3]/W,num[::3],'o',mfc='white',mec=GR,ms=4,label='full model')
ax.axhline(0.5,c=GY,ls=':',lw=0.9); ax.text(3e-2,0.515,'$b=0$ limit',color=GY,fontsize=7)
ax.set_ylim(0,0.55); ax.set_xlabel('$\\delta/W$ (dose units cancel)'); ax.set_ylabel('$\\widetilde\\chi_{\\rm LOC}/\\widetilde\\chi_{\\rm ROC}$')
ax.set_title('d  amplitude ratio',loc='left',fontsize=9); ax.legend(fontsize=6.5,frameon=False,loc='upper left',bbox_to_anchor=(0.0,0.86))
fig.savefig('fig3_boundary.pdf'); fig.savefig('fig3_boundary.png',dpi=130); plt.close(fig)

# ---------------- Supplementary: d=2 static threshold state ----------------
P=np.load('static2d.npy'); N=128; L=40.0; x=(np.arange(N)-N//2)*L/N
fig,axs=plt.subplots(1,2,figsize=(6.2,2.4)); plt.subplots_adjust(left=0.09,right=0.98,bottom=0.2,top=0.88,wspace=0.35)
im=axs[0].imshow(P,extent=[x[0],x[-1],x[0],x[-1]],cmap='viridis',origin='lower'); axs[0].set_xlim(-10,10); axs[0].set_ylim(-10,10)
plt.colorbar(im,ax=axs[0],fraction=0.046); axs[0].set_title('a  $\\Psi(\\mathbf{x})$, $d=2$',loc='left',fontsize=9)
axs[0].set_xlabel('$x$'); axs[0].set_ylabel('$y$')
ev=[-1.2303,0,0,0.5787,0.6507]
axs[1].axhline(0,c='k',lw=0.6); axs[1].plot(range(5),ev,'o',c=BL)
axs[1].annotate('unstable direction',xy=(0,-1.23),xytext=(0.8,-0.9),fontsize=7.5,color=OR,arrowprops=dict(arrowstyle='-',color=OR))
axs[1].annotate('two translation modes',xy=(1.5,0),xytext=(1.6,-0.5),fontsize=7.5,arrowprops=dict(arrowstyle='-'))
axs[1].set_xlabel('eigenvalue index'); axs[1].set_ylabel('Hessian eigenvalue'); axs[1].set_title('b  stability',loc='left',fontsize=9)
fig.savefig('figS3_static2d.pdf'); fig.savefig('figS3_static2d.png',dpi=120); plt.close(fig)
print("done")
