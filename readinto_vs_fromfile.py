import os, sys, time
import numpy as np


n = os.path.getsize('bin.m5b')
nw = n//4

tic = time.time()

#a = np.zeros(nw, dtype=np.uint32)
a = np.zeros(10**6  , dtype=np.uint32)

#with open('/dev/zero', "rb") as f:
#with open('bin2.m5b', "rb") as f:
#with open('bin.m5b', "rb") as f:

with open('rd1910_wz_268-1811.m5b', "rb") as f:
    f.readinto(a)
    
toc = time.time()

print("time readinto %6.3f" % (toc-tic))

#=====================

tic = time.time()

b = np.fromfile("rd1910_wz_268-1811.m5b", dtype=np.uint32)

toc = time.time()

print("time fromfile %6.3f" % (toc-tic))
