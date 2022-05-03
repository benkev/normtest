import math
import numpy as np
import matplotlib.pyplot as pl
# import scipy.stats

F = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))

nfrm = 1000
ndat = 2500*nfrm
thr = 0.92         # Threshold in STD 

d = np.zeros(nfrm*2500, dtype=np.uint32)   # Raw data
xt = np.zeros_like(d, dtype=np.float64)
x = np.zeros(2500, dtype=np.float64)

for ifrm in range(nfrm):
    foff = 10016*ifrm
    i0 = ifrm*2500
    i1 = i0 + 2500
    h = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, \
                    offset=foff, count=4)
    print('i0=%d, i1=%d' % (i0,i1))
    d[i0:i1] = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, \
                    offset=foff+16, count=2500)

    print('Header: 0x%08x  0x%08x  0x%08x  0x%08x' % tuple(h))

    d01 = 0x03 & d[i0:i1]     # 0th channel, bits 0 and 1
    x[np.where(d01 == 3)] =  1.5
    x[np.where(d01 == 2)] =  0.5
    x[np.where(d01 == 1)] = -0.5
    x[np.where(d01 == 0)] =  -1.5
    
    # d23 = 0b1100 & d[i0:i1]   # 1th channel, bits 0 and 1
    # x[np.where(d23 == 3)] =  1.5
    # x[np.where(d23 == 2)] =  0.5
    # x[np.where(d23 == 1)] = -0.5
    # x[np.where(d23 == 0)] =  -1.5
    
    # pl.figure()
    # pl.hist(x, rwidth=0.5, bins=[-3, -2, -1, 0, 1, 2, 3]); pl.grid(1)

d01t = 0x03 & d   # 0th channel, bits 0 and 1

xt[np.where(d01t == 3)] =  1.5
xt[np.where(d01t == 2)] =  0.5
xt[np.where(d01t == 1)] = -0.5
xt[np.where(d01t == 0)] =  -1.5

# d23t = 0b1100 & d   # 1th channel, bits 0 and 1

# xt[np.where(d23t == 3)] =  1.5
# xt[np.where(d23t == 2)] =  0.5
# xt[np.where(d23t == 1)] = -0.5
# xt[np.where(d23t == 0)] =  -1.5

pl.figure()
pl.hist(xt, rwidth=0.5, bins=[-3, -2, -1, 0, 1, 2, 3]); pl.grid(1)

# pl.show()

F = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))

Fthr = F(-thr)
# hsnor = np.array([F(-thr), F(0)-F(-thr), F(thr) - F(0), 1 - F(0.92)])
hsnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])    # Normal quantilles

hs, be = np.histogram(xt, bins=[-2, -1, 0, 1, 2])
hsrel = hs/ndat

chi2 = sum((hsnor - hsrel)**2)
        
print('Normal:       %5.3f  %5.3f  %5.3f  %5.3f' % tuple(hsnor))
print('Experimental: %5.3f  %5.3f  %5.3f  %5.3f' % tuple(hsrel))
print('Chi2: %8f' % chi2)

pl.plot([-1.5, -0.5, 0.5, 1.5], hsnor*ndat, 'ro') # Normal distribution

pl.show()
