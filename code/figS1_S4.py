# Paper: Supplementary Figures S1 (auxiliary-coordinate construction) and S4 (effective potential).
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import minimize_scalar
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
 'axes.linewidth':0.7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
BL='#0072B2'; OR='#D55E00'; GR='#009E73'; GY='0.45'
# ---------------- Fig. S1 ----------------
fig=plt.figure(figsize=(2.9,3.4)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,10); ax.set_ylim(0,12); ax.axis('off')
ax.add_patch(plt.Rectangle((1.2,7.2),6.6,3.6,fc='#FBEFE6',ec='none'))
ax.plot([1.2,7.8],[10.8,10.8],color=OR,lw=1.2); ax.plot([1.2,7.8],[7.2,7.2],color=OR,lw=1.2)
ax.add_patch(Ellipse((1.2,9.0),1.2,3.6,fc='#F6D9C6',ec=OR,lw=1.2))
ax.add_patch(Ellipse((7.8,9.0),1.2,3.6,fc='none',ec=OR,lw=1.0,ls=(0,(2,2))))
ax.plot([1.2,7.8],[7.2,7.2],color=BL,lw=3)
ax.annotate('',xy=(9.0,10.8),xytext=(9.0,7.2),arrowprops=dict(arrowstyle='<->',color=GY,lw=0.8))
ax.text(9.2,9.0,r'$2\pi R$',color=GY,va='center')
ax.text(4.5,9.0,r'$\varphi(x,y)$ on $M\times S^{1}$',ha='center',va='center',fontsize=10)
ax.text(4.5,6.5,r'$y=0$: $\Psi$ couples to $\varphi(x,0)$',ha='center',color=BL,fontsize=8.5)
ax.text(0.6,5.5,'modes seen at $y=0$:',fontsize=8.5)
for yy,lab,col,lw in [(4.6,r'$j=\pm2$: $m^{2}+4/R^{2}$',GY,1.5),(3.7,r'$j=\pm1$: $m^{2}+1/R^{2}$',GY,1.5),(2.8,r'$j=0$: $\phi_{0}$, mass$^{2}$ $m^{2}$',BL,2.5)]:
    ax.plot([0.6,2.6],[yy,yy],color=col,lw=lw); ax.text(2.9,yy,lab,va='center',fontsize=8.5,color=col if col==BL else 'k')
ax.text(0.6,1.4,r'kernel $=\coth(\pi Rq)/2q$,  $q^{2}=k^{2}+m^{2}-\omega^{2}$',fontsize=8.2)
fig.savefig('figS1_brane.pdf'); fig.savefig('figS1_brane.png',dpi=150); plt.close(fig)
# ---------------- Fig. S4 ----------------
mu,lam=1.0,1.0
V=lambda P,C,J: 0.5*(mu**2-C)*P**2+0.25*lam*P**4-J*P
P=np.linspace(-1.35,1.35,800)
fig,ax=plt.subplots(1,3,figsize=(7.0,2.1)); plt.subplots_adjust(left=0.07,right=0.99,bottom=0.2,top=0.86,wspace=0.35)
for A,(C,J,tt) in zip(ax,[(0.25,0,'a  single well, $C=0.25$'),(1.5625,0,'b  degenerate double well, $C=1.5625$'),(1.5625,0.15,'c  lifted degeneracy, $J=0.15$')]):
    A.plot(P,V(P,C,J),color=BL,lw=1.3); A.axhline(0,color='k',lw=0.4); A.set_xlabel(r'$\Psi$'); A.set_title(tt,fontsize=8.5,loc='left')
ax[0].set_ylabel(r'$V_{\mathrm{eff}}(\Psi)$')
for s in [-1,1]: ax[1].axvline(s*0.75,ls=':',color=GY,lw=0.7)
Pp=minimize_scalar(lambda x:V(x,1.5625,0.15),bounds=(0,2),method='bounded').x
Pm=minimize_scalar(lambda x:V(x,1.5625,0.15),bounds=(-2,0),method='bounded').x
dV=V(Pm,1.5625,0.15)-V(Pp,1.5625,0.15)
ins=ax[2].inset_axes([0.33,0.52,0.36,0.42])
for x0,col in [(Pm,OR),(Pp,GR)]:
    ins.plot([0,1.9],[V(x0,1.5625,0.15)]*2,color=col,lw=1.5)
ins.annotate('',xy=(0.5,V(Pp,1.5625,0.15)),xytext=(0.5,V(Pm,1.5625,0.15)),arrowprops=dict(arrowstyle='<->',lw=0.7))
ins.set_xlim(0,1.9); ins.text(1.05,(V(Pp,1.5625,0.15)+V(Pm,1.5625,0.15))/2,r'$\Delta V=%.4f$'%dV,fontsize=7,va='center')
ins.set_xticks([]); ins.tick_params(labelsize=6)
fig.savefig('figS4_potential.pdf'); fig.savefig('figS4_potential.png',dpi=150); plt.close(fig)
print(f"Delta V = {dV:.4f}")
