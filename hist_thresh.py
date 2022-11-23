import sys, os
import numpy as np
import matplotlib.pyplot as pl
import getopt, glob

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

opts, args = getopt.gnu_getopt(sys.argv[1:], '')

if len(args) == 2:
    fm5b = args[0]  # Name of *.m5b file
    ftst = args[1]  # Timestamp of *.bin files
else:
    print("Run: hist_thresh.py fname.m5b timestamp")
    raise SystemExit

fm5b_base = os.path.basename(fm5b)
fm5b_base = os.path.splitext(fm5b_base)[0]

#fquantls = glob.glob("nt_quantl_*" + fm5b_base + "*" + ftst +".bin")
fthreshs = glob.glob("nt_thresh_*" + fm5b_base + "*" + ftst +".bin")
fresidls = glob.glob("nt_residl_*" + fm5b_base + "*" + ftst +".bin")

#if len(fquantls) == 0 or len(fthreshs) == 0 or len(fresidls) == 0:
if len(fthreshs) == 0 or len(fresidls) == 0:
    print("Files *%s*.bin with the timestamp \"%s\" not found." % \
          (fm5b_base, ftst))
    raise SystemExit

#if len(fquantls) > 1 or len(fthreshs) > 1 or len(fresidls) > 1:
if len(fthreshs) > 1 or len(fresidls) > 1:
    print("Ambiguous timestamp \"%s\". Give more detail." % ftst)
    raise SystemExit

#raise SystemExit

#fquantl_name = fquantls[0]
fthresh_name = fthreshs[0]
fresidl_name = fresidls[0]

t0 = np.fromfile(fthresh_name, dtype=np.float32)
r0 = np.fromfile(fresidl_name, dtype=np.float32)

#quantl = quantl.reshape((len(quantl)//64, 16, 4)) / 2500.
thresh = t0.reshape((len(t0)//16, 16))
residl = r0.reshape((len(r0)//16, 16))

pl.figure()
pl.hist(t0, 50)
pl.title("Quantization Threshold Estimates for %s" % fm5b, fontsize=16)
pl.grid(1)
#pl.figure(); pl.hist(r0, 50); pl.grid(1)

pl.show()



