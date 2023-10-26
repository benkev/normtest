#
# plot_various_bins.py
#
# Show bin sizes and histograms for 9 quantization thresholds (+-theta)
# [0.2, 0.3, 0.5, 0.6745, 0.8, 1.0, 1.2, 1.5, 2.0].
# Assumed quantization of data samples drawn from the standard normal
# distribution.
#
#


import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf

fnorm = lambda x: (1/np.sqrt(2*np.pi))*(np.exp(-0.5*x**2))  # Normal PDF (0,1)
Fnorm = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))               # Normal CDF (0,1)

hsep = 0.3

# thr = 0.1*np.arange(1,17)
threshold = np.array([0.2, 0.3, 0.5, 0.6745, 0.8, 1.0, 1.2, 1.5, 2.0]);
xrul = np.linspace(-3., 3., 1001)
xloc = np.array([-2, -0.5, 0.5, 1.5])
f_pdf = (1/np.sqrt(2*np.pi))*(np.exp(-0.5*xrul**2))  # Normal PDF (0,1)
f_pdf = (hsep-0.05)*f_pdf/f_pdf.max()

pl.figure(figsize=(10,10))
pl.figtext(0.5, 0.91, r"Normal PDF Bins Separated by $-\infty, " \
           r"-\theta, \, 0, +\theta, +\infty$", fontsize=20, ha='center')
for ip in range(9):
    pl.subplot(3, 3, ip+1)

    thr = threshold[ip]
#    xloc =   np.array([-2, -thr[ip], thr[ip], 2])
    barloc = np.array([-2.5, -thr/2, thr/2, 2.5])
    pl.plot(xrul, f_pdf, 'orange')
    pl.fill_between(xrul, f_pdf, where=(-10<xrul)&(xrul<=-thr), color='b')
    pl.fill_between(xrul, f_pdf, where=(-thr<xrul)&(xrul<=0), color='m');
    pl.fill_between(xrul, f_pdf, where=(0<xrul)&(xrul<=thr), color='g')
    pl.fill_between(xrul, f_pdf, where=((thr<xrul)&(xrul<10)), color='c')

    xl = pl.xlim()
    pl.plot([xl[0],xl[1]], [hsep,hsep], 'k--', lw=1) # Horiz. separation line
    pl.plot([-thr,-thr], [0,0.8], 'k--', lw=0.7)   # 
    pl.plot([thr,thr], [0,0.8], 'k--', lw=0.7)     #
    pl.xticks([-2,0,2], ["$-2\sigma$", "$0$", "$2\sigma$"], fontsize=12)
    pl.yticks([])
    pl.text(-thr-0.7, 0.65, r"$-\theta$")
    pl.text(thr+0.05, 0.65, r"$+\theta$")
    
    Fthr = Fnorm(-thr)
    qnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])
    bw = 0.15 if thr < 0.4 else 0.25
    pl.bar(barloc, qnor, bw, hsep+0.05, color='red', edgecolor='k')
    pl.ylim(-0.05, 0.85)
    pl.text(xl[0], 0.78, r"$\theta=\pm%4.2f$" % thr, fontsize=12)
    #pl.grid(1)

    if ip == 3: pl.text(2.7, 0.70, "*", fontsize=30)

pl.figtext(0.5, 0.05, r"* Quantization Thresholds $\theta = " \
           "\pm 0.6745 \, \sigma$ produce histogram of  uniform distribution.",\
           fontsize=16, ha='center')


pl.show()

