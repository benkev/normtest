#
# inspect_nt.py
#
# python inspect_nt <m5b_filename> <nt_file_timestamp> <start_frame#> \
#          [<end_frame#>]         
#
#


import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
import getopt, glob

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

opts, args = getopt.gnu_getopt(sys.argv[1:], '')

fm5b = args[0]
ftst = args[1]

fm5b_base = os.path.basename(fm5b)
fm5b_base = os.path.splitext(fm5b_base)[0]

fquantls = glob.glob("nt_quantl_*" + fm5b_base + "*" + ftst +".bin")
fthreshs = glob.glob("nt_thresh_*" + fm5b_base + "*" + ftst +".bin")

if len(fquantl) > 1 or len(fthresh) > 1:
    print("Ambiguous timestamp. Give more detail.")
    raise SystemExit

fquantl_name = fquantls[0]
fthresh_name = fthreshs[0]

fquantl = open(fquantl_name, "rb")
fthresh = open(fthresh_name, "rb")


fquantl.close()
fthresh.close()









