import os, sys
import getopt, glob
import numpy as np
import matplotlib.pyplot as pl
import scipy.stats


opts, args = getopt.gnu_getopt(sys.argv[1:], '')

fm5b = args[0]
ftst = args[1]
frm0 = int(args[2])
n_frms = 1

fm5b_base = os.path.basename(fm5b)
fm5b_base = os.path.splitext(fm5b_base)[0]

fquantls = glob.glob("nt_quantl_*" + fm5b_base + "*" + ftst +".bin")

if len(fquantls) == 0: # or len(fthreshs) == 0 or len(fchi2s) == 0:
    print("Files *%s*.bin with the timestamp \"%s\" not found." % \
          (fm5b_base, ftst))
    raise SystemExit

if len(fquantls) > 1: # or len(fthreshs) > 1 or len(fchi2s) > 1:
    print("Ambiguous timestamp \"%s\". Give more detail." % ftst)
    raise SystemExit
fquantl_name = fquantls[0]
#
# Find file offsets to read the data from
#
cnt_q = 16*4*n_frms          # float32 words for quantiles
offs_qdat = 16*4*4*frm0      # Offset bytes for quantiles

quantl = np.fromfile(fquantl_name, dtype=np.float32, \
                      offset=offs_qdat, count=cnt_q)

qu = quantl.reshape((16, 4))

#
# Expectations of 25%-Quantiles of Normal PDF N(0,1)
#
# Quantile (-Inf .. -0.6745): -mu[1] = -1.2711140638164977
# Quantile (-0.6745 .. 0):    -mu[0] = -0.324667388612524
# Quantile (0 .. 0.6745):      mu[0] =  0.324667388612524 
# Quantile (0.6745 .. +Inf):   mu[1] =  1.2711140638164977
#
mu = np.array([0.324667388612524, 1.2711140638164977])

N = 2500

sk = np.sqrt(N)*(mu[0]**3*(qu[:,2]**3 - qu[:,1]**3) +       \
                 mu[1]**3*(qu[:,3]**3 - qu[:,0]**3)) /      \
                 (mu[0]**2*(qu[:,1]**2 + qu[:,2]**2) +       \
                  mu[1]**2*(qu[:,0]**2 + qu[:,3]**2))**1.5


print("sk[0..15] =")
print(sk)






