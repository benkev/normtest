#
# ex4_1_chi2.py
#
# 
#

import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf

fnorm = lambda x, u, s: (1/np.sqrt(2*np.pi))*(np.exp(-0.5*((x - u)/s)**2))
Fnorm = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))               # Normal CDF (0,1)

N = np.float32(100)

mup = np.float32(20.0)
sigp = np.float32(0.5)

mus = np.float32(19.94)         # = sum(ldat*hh)/100
sigs = np.float32(0.52)         # = sqrt(sum(hh*ldat**2)/100 - mus**2)

ldat = np.array([18.7, 18.9, 19.1, 19.3, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5,
                 20.7, 20.9, 21.1, 21.3], dtype=np.float32)
hh = np.array([1, 3, 4, 7, 13, 14, 11, 12, 16, 11, 4, 1, 1, 2],
              dtype=np.float32)
yp = np.array([], dtype=np.float32)
