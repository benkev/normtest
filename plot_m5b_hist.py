#
# plot_m5b_hist.py
#

import sys, os
import numpy as np
import matplotlib.pyplot as pl
import getopt, glob

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

opts, args = getopt.gnu_getopt(sys.argv[1:], '')

fm5b = args[0]
ftst = args[1]


fm5b_base = os.path.basename(fm5b)
fm5b_base = os.path.splitext(fm5b_base)[0]

fquantls = glob.glob("nt_quantl_*" + fm5b_base + "*" + ftst +".bin")
fthreshs = glob.glob("nt_thresh_*" + fm5b_base + "*" + ftst +".bin")
fchi2s = glob.glob("nt_chi2_*" + fm5b_base + "*" + ftst +".bin")

if len(fquantls) == 0 or len(fthreshs) == 0 or len(fchi2s) == 0:
    print("Files *%s*.bin with the timestamp \"%s\" not found." % \
          (fm5b_base, ftst))
    raise SystemExit

if len(fquantls) > 1 or len(fthreshs) > 1 or len(fchi2s) > 1:
    print("Ambiguous timestamp \"%s\". Give more detail." % ftst)
    raise SystemExit

fquantl_name = fquantls[0]
fthresh_name = fthreshs[0]
fchi2_name = fchi2s[0]


c2 = np.fromfile(fchi2_name, dtype=np.float32)
th = np.fromfile(fthresh_name, dtype=np.float32)
qu = np.fromfile(fquantl_name, dtype=np.float32)

pl.figure(figsize=(6,7))
pl.subplot(2,1,1)
pl.title(fm5b)
pl.grid(1)
pl.hist(c2, 100, label="$\chi^2$")
pl.xlabel(r"$\chi^2$")
pl.legend()

pl.subplot(2,1,2)
pl.grid(1)
pl.hist(th, 100, label="$\theta$")
pl.xlabel(r"$\theta$")
pl.legend()


pl.show()
