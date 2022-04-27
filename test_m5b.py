import numpy as np
import matplotlib.pyplot as pl
# import scipy.stats

h = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, offset=0, \
                count=4)
d = np.fromfile('rd1910_wz_268-1811.m5b', dtype=np.uint32, offset=16, \
                count=2500)

print('Header:')
for i in range(4): print('0x%08x' % h[i])

d01 = 0x03 & d

x = np.zeros_like(d01, dtype=np.float64)

x[np.where(d01 == 3)] =  1.5
x[np.where(d01 == 2)] =  0.5
x[np.where(d01 == 1)] = -0.5
x[np.where(d01 == 0)] =  -1.5

pl.figure()
pl.hist(x, rwidth=0.5, bins=[-3, -2, -1, 0, 1, 2, 3]); pl.grid(1)

pl.show()
