#
# Plot squared error (the residual) between M5B Data and four quantiles of the
# standard normal distribution over the range [0.4 .. 1.5] of input sample
# thresholds. The plot obviously has its minimum at the threshold +-0.817/STD
# marked with the red dot.
#
# 
#

import math
import numpy as np
import matplotlib.pyplot as pl


F = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))

def calc_hserr(xt, thr, prres=False):
    """
    Calculate error between the normal quantile histograms of the experimental
    signal data xt and the corresponding quantiles of the normal distribution.
    The quantiles are for [-inf..-thr], [-thr..0], [0..thr] [thr..+inf].
    """
    Fthr = F(-thr)
    # hsnor = np.array([F(-thr), F(0)-F(-thr), F(thr) - F(0), 1 - F(0.92)])
    hsnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])  # Normal quantiles

    hs, be = np.histogram(xt, bins=[-2, -1, 0, 1, 2])
    hsrel = hs/ndat

    chi2 = sum((hsnor - hsrel)**2)

    if prres:
        print('Normal:       %5.3f  %5.3f  %5.3f  %5.3f' % tuple(hsnor))
        print('Experimental: %5.3f  %5.3f  %5.3f  %5.3f' % tuple(hsrel))
        print('Chi2: %8f' % chi2)
    return chi2, hsnor, hsrel


def calc_chi2(xt, thr):
    """
    Calculate error between the normal quantile histograms of the experimental
    signal data xt and the corresponding quantiles of the normal distribution.
    The quantiles are for [-inf..-thr], [-thr..0], [0..thr] [thr..+inf].
    """
    ndat = len(xt)
    
    Fthr = F(-thr)

    E = ndat*np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr]) 

    B, be = np.histogram(xt, bins=[-2, -1, 0, 1, 2])

    chi2 = sum((E - B)**2/E)
    
    return chi2


    
pl.ion()
pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

nthr = 51
nfrm = 1  # 000
ndat = 2500*nfrm
thrs = np.linspace(0.4, 1.5, nthr)       # Thresholds in STD
chi2s = np.zeros(nthr,  dtype=np.float64)

d = np.zeros(nfrm*2500, dtype=np.uint32)   # Raw data
xt = np.zeros_like(d, dtype=np.float64)
x = np.zeros(2500, dtype=np.float64)

for ifrm in range(nfrm):
    foff = 10016*ifrm
    i0 = ifrm*2500
    i1 = i0 + 2500
    h = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, \
                    offset=foff, count=4)
    d[i0:i1] = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, \
                    offset=foff+16, count=2500)

d01t = 0x03 & d   # 0th channel, bits 0 and 1

xt[np.where(d01t == 3)] =  1.5
xt[np.where(d01t == 2)] =  0.5
xt[np.where(d01t == 1)] = -0.5
xt[np.where(d01t == 0)] =  -1.5


for ithr in range(nthr):
    thr = thrs[ithr]

    chi2 = calc_chi2(xt, thr)

    chi2s[ithr] = chi2

thr_opt = 0.7988   # Optimal threshold
chi2_opt = calc_chi2(xt, thr_opt)

pl.figure(); pl.plot(thrs, chi2s); pl.grid(1)
pl.plot(thr_opt, chi2_opt, 'ro')

yl = pl.ylim()
ytxt = yl[0] + (yl[1] - yl[0])/2

# pl.title(r'Error b/w M5B Data and Normal Quantiles ' \
#          'for Tresholds [0.4 .. 1.5] ', fontsize=16)
pl.figtext(0.5, 0.93, r'Error b/w M5B Data and Normal Quantiles ' \
           'for Tresholds [0.4 .. 1.5] ', ha='center', fontsize=15)
pl.xlabel(r'$\theta=$ (quantization threshold)/$\sigma$', fontsize=14)
pl.ylabel(r'Error, $\epsilon^2$', fontsize=14)

pl.text(0.7, ytxt, r'Optimum:', fontsize=16)
pl.text(0.57, ytxt-150, r'$\theta=%5.3f, \, \chi^2=%8.6f$' % \
                        (thr_opt, chi2_opt), fontsize=16)
pl.show()
