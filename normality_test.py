#
# normality_test.py
#

import numpy as np
import matplotlib.pyplot as pl

from numpy.random import seed
from numpy.random import normal

# seed(1)
# dat = normal(loc=0, scale=1, size=int(1e7))
# pl.figure(); pl.hist(dat, bins=101) 

# pl.show()
# pl.grid(1)

h = 3.0
x = 9 + h*np.arange(9)
frq = np.array([2, 6, 10, 17, 33, 11, 9, 7, 5], dtype=float)
N = np.sum(frq)

x2n = (x**2)*frq
xmean = np.sum(x*frq)/N
xsig = np.sqrt(np.sum((x**2)*frq)/N - xmean**2)
z = (x - xmean)/xsig  # Standardized 
f = np.exp(-z**2/2)/np.sqrt(2*np.pi) # Standard normal PDF
tfrq = (h*N/xsig)*f  # Theoretical frequency
chi2 = np.sum((frq - tfrq)**2/tfrq)


pl.figure();
pl.plot(x, frq, 'o'); pl.grid(1)
pl.plot(x, frq)
pl.plot(x, tfrq)
 
pl.show()


