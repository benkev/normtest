#
# chi2.py
#
# 
#

import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf

N = 2500       # Measurements in one m5b data frame (per channel)
Nf = 2000      # Number of frames to read
nq = 16*4      # Words in quantl file: 16 chans x 4 quantiles per 1 frame
nt = 16        # Words in thresh file: 16 chans per 1 frame
cnt = Nf*nt   # Number of np.float32 words to read from thresh
offt = 0*nt*4  # Offset in bytes from thresh
cnq = Nf*nq   # Number of np.float32 words to read from quantl
offq = 0*nq*4  # Offset in bytes from quantl

fbase = "rd1910_wz_268-1811cuda_"


fnorm = lambda x: (1/np.sqrt(2*np.pi))*(np.exp(-0.5*x**2))  # Normal PDF (0,1)
Fnorm = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))               # Normal CDF (0,1)

q0 = np.fromfile(fbase + "quantl.bin", dtype=np.float32, offset=offq, count=cnq)
t0 = np.fromfile(fbase + "thresh.bin", dtype=np.float32, offset=offt, count=cnt)

qua = q0.reshape((len(q0)//64, 16, 4))   # / N
thr = t0.reshape((len(t0)//16, 16))

Fthr = Fnorm(-thr)
qnor = np.zeros((Nf,16,4), dtype=np.float32)
qnor[:,:,0] = qnor[:,:,3] = Fthr
qnor[:,:,1] = qnor[:,:,2] = 0.5-Fthr

c2 = np.sum((qua - N*qnor)**2 / qua, axis=2)

pl.figure(); pl.hist(c2.flatten(), 100); pl.grid(1)
pl.xlabel(r"$\chi^2 $")

pl.show()
