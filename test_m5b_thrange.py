import math
import numpy as np
import matplotlib.pyplot as pl

# pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

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

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

nthr = 51
nfrm = 1000
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
    # for ifrm in range(nfrm):
    #     foff = 10016*ifrm
    #     i0 = ifrm*2500
    #     i1 = i0 + 2500
    #     h = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, \
    #                     offset=foff, count=4)
    #     d[i0:i1] = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, \
    #                     offset=foff+16, count=2500)

    # d01t = 0x03 & d   # 0th channel, bits 0 and 1

    # xt[np.where(d01t == 3)] =  1.5
    # xt[np.where(d01t == 2)] =  0.5
    # xt[np.where(d01t == 1)] = -0.5
    # xt[np.where(d01t == 0)] =  -1.5

    chi2, hsnor, hsrel = calc_hserr(xt, thr, prres=False)

    chi2s[ithr] = chi2

thr_opt = 0.817   # Optimal threshold
chi2_opt, hsnor_opt, hsrel_opt = calc_hserr(xt, thr_opt, prres=True)

pl.figure(); pl.plot(thrs, chi2s); pl.grid(1)
pl.plot(thr_opt, chi2_opt, 'ro')
pl.title(r'Error b/w M5B Data and Normal Quantiles ' \
         'for Tresholds [0.4 .. 1.5] ', fontsize=14)
pl.xlabel(r'$\alpha=$ (quantization threshold)/$\sigma$', fontsize=14)
pl.ylabel(r'Error, $\epsilon^2$', fontsize=14)

pl.text(0.7, 0.036, r'Optimum:', fontsize=12)
pl.text(0.62, 0.031, r'$\alpha=%5.3f, \, \epsilon^2=%8.6f$' % \
        (thr_opt, chi2_opt), fontsize=12)

pl.show()
