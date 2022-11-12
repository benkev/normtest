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

if len(args) > 3:
    frm1 = int(args[3])
else:
    frm1 = None



fm5b_base = os.path.basename(fm5b)
fm5b_base = os.path.splitext(fm5b_base)[0]

fquantls = glob.glob("nt_quantl_*" + fm5b_base + "*" + ftst +".bin")
fthreshs = glob.glob("nt_thresh_*" + fm5b_base + "*" + ftst +".bin")
fresidls = glob.glob("nt_residl_*" + fm5b_base + "*" + ftst +".bin")

if len(fquantls) == 0 or len(fthreshs) == 0 or len(fresidls) == 0:
    print("Files *%s*.bin with the timestamp \"%s\" not found." % \
          (fm5b_base, ftst))
    raise SystemExit

if len(fquantls) > 1 or len(fthreshs) > 1 or len(fresidls) > 1:
    print("Ambiguous timestamp \"%s\". Give more detail." % ftst)
    raise SystemExit

#raise SystemExit

fquantl_name = fquantls[0]
fthresh_name = fthreshs[0]
fresidl_name = fresidls[0]

#
# Find file offsets to read the data from
#
if frm1:
    n_frms = frm1 - frm0
else:
    n_frms = 1

cnt_q = 16*4*n_frms          # float32 words
offs_qdat = 16*4*4*frm0      # Bytes

cnt_t = 16*n_frms            # float32 words
cnt_r = cnt_t                # float32 words
offs_tdat = 16*4*frm0        # Bytes
offs_rdat = offs_tdat        # Bytes

quantl = np.fromfile(fquantl_name, dtype=np.float32, \
                      offset=offs_qdat, count=cnt_q)
thresh = np.fromfile(fthresh_name, dtype=np.float32, \
                      offset=offs_tdat, count=cnt_t)

residl = np.fromfile(fresidl_name, dtype=np.float32, \
                      offset=offs_rdat, count=cnt_t)
if frm1:
    quantl = quantl.reshape((len(quantl)//64, 16, 4)) / 2500.
    thresh = thresh.reshape((len(thresh)//16, 16))
    residl = residl.reshape((len(residl)//16, 16))
else:
    quantl = quantl.reshape((16, 4)) / 2500.
    # thresh = thresh.reshape((16))
    # residl = residl.reshape((16))


F = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))  # Normal CDF

F_thr = F(-thresh)

if frm1:
    quanor = np.zeros((n_frms, 16, 4), dtype=np.float32)
    quanor[:,:,0] = quanor[:,:,3] = F_thr
    quanor[:,:,1] = quanor[:,:,2] = 0.5 - F_thr
else:
    quanor = np.zeros((16, 4), dtype=np.float32)
    quanor[:,0] = quanor[:,3] = F_thr
    quanor[:,1] = quanor[:,2] = 0.5 - F_thr

    resnor = np.zeros(16, dtype=np.float32)
    
    for ich in range(16):
        resnor[ich] = np.sum((quanor[ich,:] - quantl[ich,:])**2)

    # for ich in range(16):
    #     pl.figure()
    #     pl.plot(quantl[ich,:], 'b'); pl.plot(quanor[ich,:], 'r')
    #     pl.grid(1)

pl.show()
