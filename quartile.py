#
# quartile.py
#
# Show quartiles for 
#
#


import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf

fnorm = lambda x: (1/np.sqrt(2*np.pi))*(np.exp(-0.5*x**2))  # Normal PDF (0,1)
Fnorm = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))               # Normal CDF (0,1)

sepl = 0.3

# thr = 0.1*np.arange(1,17)
thr = np.array([0.2, 0.3, 0.5, 0.67, 0.8, 1.0, 1.2, 1.5, 2.0]);
xrul = np.linspace(-3., 3., 1001)
xloc = np.array([-2, -0.5, 0.5, 1.5])
f_pdf = (1/np.sqrt(2*np.pi))*(np.exp(-0.5*xrul**2))  # Normal PDF (0,1)
f_pdf = (sepl-0.05)*f_pdf/f_pdf.max()

pl.figure(figsize=(10,10))
pl.figtext(0.5, 0.95, "Normal PDF Quartiles Separated by $-\infty$, -thr, 0, " \
           "thr, $+\infty$", fontsize=16, ha='center')
for ip in range(9):
    pl.subplot(3, 3, ip+1)

#    xloc =   np.array([-2, -thr[ip], thr[ip], 2])
    barloc = np.array([-2.5, -thr[ip]/2, thr[ip]/2, 2.5])
    pl.plot(xrul, f_pdf, 'r')
    pl.fill_between(xrul, f_pdf, where=(-10<xrul)&(xrul<=-thr[ip]), color='b')
    pl.fill_between(xrul, f_pdf, where=(-thr[ip]<xrul)&(xrul<=0), color='m');
    pl.fill_between(xrul, f_pdf, where=(0<xrul)&(xrul<=thr[ip]), color='g')
    pl.fill_between(xrul, f_pdf, where=((thr[ip]<xrul)&(xrul<10)), color='c')

    xl = pl.xlim()
    pl.plot([xl[0],xl[1]], [sepl,sepl], 'k--', lw=1) # Horiz. separation line
    
    Fthr = Fnorm(-thr[ip])
    qnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])
    pl.bar(barloc, qnor, 0.15, sepl+0.05, color='red', edgecolor='k')
    pl.ylim(-0.05, 0.85)
    pl.text(xl[0]+0.05, 0.78, "thr=%3.1f" % thr[ip])
    pl.grid(1)
        
pl.show()

