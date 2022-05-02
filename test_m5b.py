import numpy as np
import matplotlib.pyplot as pl
# import scipy.stats

nfrm = 1000

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

    d01 = 0x03 & d[i0:i1]   # 0th channel, bits 0 and 1
    x[np.where(d01 == 3)] =  1.5
    x[np.where(d01 == 2)] =  0.5
    x[np.where(d01 == 1)] = -0.5
    x[np.where(d01 == 0)] =  -1.5
    
    # pl.figure()
    # pl.hist(x, rwidth=0.5, bins=[-3, -2, -1, 0, 1, 2, 3]); pl.grid(1)

d01t = 0x03 & d   # 0th channel, bits 0 and 1

xt[np.where(d01t == 3)] =  1.5
xt[np.where(d01t == 2)] =  0.5
xt[np.where(d01t == 1)] = -0.5
xt[np.where(d01t == 0)] =  -1.5


pl.figure()
pl.hist(xt, rwidth=0.5, bins=[-3, -2, -1, 0, 1, 2, 3]); pl.grid(1)

pl.show()
