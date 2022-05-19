import math
import numpy as np
import matplotlib.pyplot as pl
import time
from fminbound import fminbound


F = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

F = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))  # Normal CDF

def quanerr(thr, args):
    Fthr = F(-thr)
    hsnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])  # Normal quantiles
    hsexp = np.copy(args)
    err = sum((hsnor - hsexp)**2)
    return err

t_start = time.time()


nfrm = 100000
ndat = 2500*nfrm

d = np.zeros(nfrm*2500, dtype=np.uint32)   # Raw data
xt = np.zeros_like(d, dtype=np.float64)
x = np.zeros(2500, dtype=np.float64)

#
# Read nfrm frames into array d
#
for ifrm in range(nfrm):
    foff = 10016*ifrm
    i0 = ifrm*2500
    i1 = i0 + 2500
    h = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, \
                    offset=foff, count=4)
    d[i0:i1] = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, \
                    offset=foff+16, count=2500)

t_read = time.time() - t_start
print("--- read: %6f seconds ---" % t_read)
t_read = time.time()


d01t = 0x03 & d   # 0th channel, bits 0 and 1

xt[np.where(d01t == 3)] =  1.5
xt[np.where(d01t == 2)] =  0.5
xt[np.where(d01t == 1)] = -0.5
xt[np.where(d01t == 0)] =  -1.5

t_where = time.time() - t_read
print("--- where: %6f seconds ---" % t_where)
t_where = time.time()

hs, be = np.histogram(xt, bins=[-2, -1, 0, 1, 2])  # Experimental quantiles
hsexp = hs/ndat

t_hist = time.time() - t_where
print("--- hist: %6f seconds ---" % t_hist)
t_hist = time.time()

thr = fminbound(quanerr, args=(hsexp,), bounds=[0.5,1.2], disp=3)

t_topt = time.time() - t_hist
print("--- optimization: %6f seconds ---" % t_topt)

Fthr = F(-thr)
hsnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])  # Normal quantiles

err = sum((hsnor - hsexp)**2)

print('Normal:       %5.3f  %5.3f  %5.3f  %5.3f' % tuple(hsnor))
print('Experimental: %5.3f  %5.3f  %5.3f  %5.3f' % tuple(hsexp))
print('Threshold: %8f  Error: %8f' % (thr, err))

