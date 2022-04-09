#
# normality_test.py
#

import numpy as np
import matplotlib.pyplot as pl
import scipy.stats

from numpy.random import seed
from numpy.random import normal

seed(1)
dat = normal(loc=0.0, scale=1.0, size=int(1e8))
#dat = normal(loc=2.4, scale=3.7, size=int(1e3))

pl.figure(); pl.hist(dat, bins=20) 
pl.show()
pl.grid(1)

h =1.0
lx = np.floor(dat.min()) # Left interval limit
rx = np.ceil(dat.max())  # Right interval limit
bins = np.arange(lx, rx+1)     # Bin edges
x = 0.5 + np.arange(lx, rx)  # Bin middle points
frq, binedg = np.histogram(dat, bins=bins)

N = np.sum(frq)

x2n = (x**2)*frq
xmean = np.sum(x*frq)/N
xsig = np.sqrt(np.sum((x**2)*frq)/N - xmean**2)
z = (x - xmean)/xsig  # Standardized 
f = np.exp(-z**2/2)/np.sqrt(2*np.pi) # Standard normal PDF
tfrq = (h*N/xsig)*f  # Theoretical frequency
chi2 = np.sum((frq - tfrq)**2/tfrq)

#
# Critical value of chi2(k) with 10 degrees of freedom, for significance 0.95
#
# dat is normal (at significance 0.95) if chi2 < chi2cr
#
k = len(x) - 2 - 1
chi2cr = scipy.stats.chi2.ppf(1-0.95, k)

pl.figure();
pl.plot(x, frq, 'o'); pl.grid(1)
pl.plot(x, frq)
pl.plot(x, tfrq)
 
pl.show()

cmps = ' < ' if chi2 < chi2cr else ' > '
norm_or_not = ' -- Normal ' if chi2 < chi2cr else ' -- NOT Normal '
print('chi2 = %6.2f' % chi2, ', chi2cr = %6.2f' % chi2cr)
print('chi2_observed' + cmps + 'chi2_critical: ' + \
      norm_or_not + ' at 0.95 significance level')
