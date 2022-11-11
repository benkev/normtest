#
# inspect_nt.py
#
# python inspect_nt.py <m5b_filename> <nt_file_timestamp> <start_frame#> \
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
frm0 = int(args[2])
frm1 = int(args[3])



fm5b_base = os.path.basename(fm5b)
fm5b_base = os.path.splitext(fm5b_base)[0]

fquantls = glob.glob("nt_quantl_*" + fm5b_base + "*" + ftst +".bin")
fthreshs = glob.glob("nt_thresh_*" + fm5b_base + "*" + ftst +".bin")

if len(fquantls) == 0 or len(fthreshs) == 0:
    print("File *%s*.bin with the timestamp \"%s\" not found." % \
          (fm5b_base, ftst))
    raise SystemExit

if len(fquantls) > 1 or len(fthreshs) > 1:
    print("Ambiguous timestamp \"%s\". Give more detail." % ftst)
    raise SystemExit

#raise SystemExit

fquantl_name = fquantls[0]
fthresh_name = fthreshs[0]

#
# Find file offsetts to read the data from
#
n_frms = frm1 - frm0

cnt_q = 16*4*n_frms          # float32 words
offs_qdat = 16*4*4*frm0 + 16 # Bytes

cnt_t = 16*n_frms            # float32 words
offs_tdat = 16*4*frm0 + 16   # Bytes

fquantl = np.fromfile(fquantl_name, dtype=np.float32, \
                      offset=offs_qdat, count=cnt_q)
fthresh = np.fromfile(fthresh_name, dtype=np.float32, \
                      offset=offs_tdat, count=cnt_t)










