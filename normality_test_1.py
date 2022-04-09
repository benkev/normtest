#
# normality_test.py
#

import numpy as np
import matplotlib.pyplot as pl
import scipy.stats

from numpy.random import seed
from numpy.random import normal

N = int(1e5)
alpha = 0.95

# seed(12)
dat = normal(loc=0.0, scale=1.0, size=N)
#dat = normal(loc=2.4, scale=3.7, size=int(1e3))

# pl.figure(); pl.hist(dat, bins=20) 
# pl.show()
# pl.grid(1)

h = 0.25                  # Interval length
#h = 1.0                   # Interval length
#h = 0.125                 # Interval length
lx = np.floor(dat.min()) # Left interval limit
rx = np.ceil(dat.max())  # Right interval limit
nedges = int((rx - lx)/h) + 1
edges = np.linspace(lx, rx, nedges)    # Bin edges
x = edges[:-1] + 0.5*h
# x = h*(0.5 + np.arange(lx, rx))  # Bin middle points
frq, binedg = np.histogram(dat, bins=edges)
nfrq = len(frq)

# raise SystemExit

xmean = np.sum(x*frq)/N
xsig = np.sqrt(np.sum((x**2)*frq)/N - xmean**2)
z = (x - xmean)/xsig  # Standardized 
f = np.exp(-z**2/2)/np.sqrt(2*np.pi) # Standard normal PDF
tfrq = (h*N/xsig)*f  # Theoretical frequency
chi2 = np.sum((frq - tfrq)**2/tfrq)

#
# Critical value of chi2(k) with k degrees of freedom, for significance 0.95
#
# dat is normal (at significance alpha) if chi2 < chi2cr
#
k = nfrq - 2 - 1
chi2cr = scipy.stats.chi2.ppf(alpha, k)

pl.figure();
pl.plot(x, frq, 'o'); pl.grid(1)
pl.plot(x, frq)
pl.plot(x, tfrq)
 
pl.show()

cmps = ' < ' if chi2 < chi2cr else ' > '
norm_or_not = ' -- Normal ' if chi2 < chi2cr else ' -- NOT Normal '
print('N = ', N, ', nfrq = ', nfrq)
print('chi2_observed = %6.2f' % chi2, ', chi2_critical = %6.2f' % chi2cr, \
      ' at k = %d degrees of freedom' % k)
print('chi2_observed' + cmps + 'chi2_critical: ' + \
      norm_or_not + ' at %4.2f significance level' % alpha)
