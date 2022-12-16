#
# chi2.py
#
# 
#

import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
from scipy.stats import chi2

N = 2500       # Measurements in one m5b data frame (per channel)
Nf = 20000     # Number of frames to read
nq = 16*4      # Words in quantl file: 16 chans x 4 quantiles per 1 frame
nt = 16        # Words in thresh file: 16 chans per 1 frame
cnt = Nf*nt    # Number of np.float32 words to read from thresh
cnq = Nf*nq    # Number of np.float32 words to read from quantl
off = 0        # Offset in elements of quantl and thresh
offt = off*nt*4  # Offset in bytes for thresh
offq = off*nq*4  # Offset in bytes for quantl

fbase = "rd1910_wz_268-1811cuda_"
#fbase = "rd1903_ft_100-0950"

fnorm = lambda x: (1/np.sqrt(2*np.pi))*(np.exp(-0.5*x**2))  # Normal PDF (0,1)
Fnorm = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))               # Normal CDF (0,1)

q0 = np.fromfile(fbase + "quantl.bin",dtype=np.float32,offset=offq, count=cnq)
t0 = np.fromfile(fbase + "thresh.bin",dtype=np.float32,offset=offt, count=cnt)

# q0 = np.fromfile("bin_leonid2/nt_quantl_cuda_" + fbase +
#                  "_20221115_155722.652.bin",
#                  dtype=np.float32, offset=offq, count=cnq)
# t0 = np.fromfile("bin_leonid2/nt_thresh_cuda_" + fbase +
#                  "_20221115_155722.652.bin",
#                  dtype=np.float32, offset=offt, count=cnt)

qobs = q0.reshape((len(q0)//64, 16, 4))   # Observed quantiles
thr = t0.reshape((len(t0)//16, 16))

#
# Quantiles of Normal distribution
#
Fthr = Fnorm(-thr)
qnor = np.zeros((Nf,16,4), dtype=np.float32)
qnor[:,:,0] = qnor[:,:,3] = Fthr
qnor[:,:,1] = qnor[:,:,2] = 0.5-Fthr

c2 = np.sum((qobs - N*qnor)**2 / qobs, axis=2)    # 
c2n = np.sum((qobs - N*qnor)**2 / N*qnor, axis=2)

#
# Degrees of freedom, df
#
# For each frame we have 4 random values. They are bound by 2 restrictions:
# (1) Normalizing factor N = 2500, and
# (2) Quantization threshold found from the observed numbers.
# Therefore, the number of degrees of freedom for chi^2 distribution
# is 2 = 4 - 2.
#
df = 2

c2r = c2/df   # Reduced chi^2

x = np.linspace(0, 10, 100)

pl.figure();
pl.plot(x, chi2.pdf(x, 2), label="PDF $\chi^2_2(x)$"); pl.grid(1)
pl.plot(x, chi2.cdf(x, 2), label="CDF $\chi^2_2(x)$");
pl.legend()

pl.figure(); pl.hist(c2r.flatten(), 100); pl.grid(1)
pl.xlabel(r"Reduced $\chi^2_\nu = \chi^2$/$\mathrm{df}$")


pl.figure(); pl.plot(x, 1 - chi2.cdf(x, 2)); pl.grid(1)
pl.title(r"$1 - \mathrm{CDF}[\chi^2_\nu(2,x)]$")
pl.xlabel(r"$x$")

#
# chi2.ppf and chi2.cdf are inverse of each-other:
#
# from scipy.stats.distributions import chi2
# chi2.ppf(0.95, df=5)     # 11.07
# chi2.cdf(11.07, df=5)    # 0.95
#
# chi2.isf(q, *args, **kwds)
# Inverse survival function (inverse of `sf`) at q of the given RV (Random
# Variable)
#
# from scipy import stats
# upperv=stats.chi2.isf(1-alpha/2,nu)
#
# Quantile corresponding to the upper tail probability q.
#


y = np.logspace(-3, 0, 100)
pl.figure(); pl.plot(y, chi2.ppf(y, 2)); pl.grid(1)
pl.title(r"$\mathrm{PPF}[\chi^2_\nu(2,x)]$")
pl.xlabel(r"$x$")

pval = np.array([0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10,
                 0.05, 0.01, 0.001])

for p in pval:
    #print(1-p, chi2.ppf(p, df=2))
    print("%f  %f" % (p, chi2.isf(p, df=2)))
print()

for p in pval:
    print("%f  %f" % (p, chi2.ppf(1-p, df=2)))
    #print("%f  %f" % (p, chi2.isf(p, df=4)))

pl.show()

# print("Critical value of $\chi^2_2(x)$ for $p_{value}= 0.05$: %f" % \
#       chi2.isf(p, df=2))
print("Critical value of chi^2(2,x) for p-value = 0.05: %f" % \
      chi2.isf(p, df=2))



