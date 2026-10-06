"""Paper: Section 4.2 (limitations, homogeneity and isotropy) and Supplementary S6.
Noise-free test of the isotropic P1/P4 analysis on an anisotropic sheet: both propagation speeds
are scaled by (1+eps) along x and (1-eps) along y. Fields are computed exactly by 2D FFT on a
periodic 768 x 768 grid (spacing 0.1), sampled at h = 0.2 over a patch of half-width 6, and
analysed with the isotropic routes of recovery_sheet.py."""
import numpy as np
from scipy.special import j0
import recovery_sheet as RS
Nf,dx=768,0.1; x=(np.arange(Nf)-Nf//2)*dx; kx=2*np.pi*np.fft.fftfreq(Nf,dx); KX,KY=np.meshgrid(kx,kx,indexing='ij')
S2=np.exp(-0.5*(KX**2+KY**2)*RS.SIG**2)
step=int(round(0.2/dx)); n=int(round(6/0.2)); idx=Nf//2+np.arange(-n,n+1)*step
X,Y=np.meshgrid(x[idx],x[idx],indexing='ij'); r=np.hypot(X,Y).ravel()
ru,inv=np.unique(np.round(r,9),return_inverse=True)
kfit=np.linspace(2*np.pi/12,min(np.pi/0.2,12.0),40); J0k=j0(np.outer(kfit,r))*0.2**2
kq=np.linspace(0,60,6001); Gq=RS.Gtk(RS.th0,kq,RS.t)          # G(t,k') on a fine k' grid
c1,c2,g1,g2,a,m2,gg=RS.th0
for eps in [0.0,0.05,0.1,0.2]:
    kp=np.sqrt((1+eps)**2*KX**2+(1-eps)**2*KY**2)
    U=np.empty((len(r),len(RS.t)))
    for it in range(len(RS.t)):
        Gk=np.interp(kp,kq,Gq[:,it]); f=np.real(np.fft.ifft2(Gk*S2))/dx**2
        U[:,it]=np.fft.fftshift(f)[np.ix_(idx,idx)].ravel()
    chi0=(c2**2*kp**2+m2)/((c1**2*kp**2+a)*(c2**2*kp**2+m2)-gg)
    us=(np.fft.fftshift(np.real(np.fft.ifft2(chi0*S2))/dx**2)[np.ix_(idx,idx)]).ravel()
    th,_=RS.fit_P1(J0k@U,kfit); k1,_,R1=RS.statics(th)
    k4,R4=RS.fit_P4(us,lambda kap: RS.kernel_profile(ru,kap)[inv],RS.KAP0,RS.B0)
    print(f"eps={eps:.2f}: P1 kappa={np.round(k1,4)} R-={R1[0]:.3f} | P4 kappa={np.round(k4,4)} R-={R4[0]:.3f} | "
          f"route mismatch {np.round((k1/k4-1)*100,2)}% | P1 vs isotropic truth {np.round((k1/RS.KAP0-1)*100,2)}%")
