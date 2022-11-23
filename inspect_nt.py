#
# inspect_nt.py
#
# python inspect_nt.py <m5b_filename> <nt_file_timestamp> <start_frame#> \
#          [<n_frames>]         
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
    n_frms = int(args[3])   # Number of frames is given on the cmd line
else:
    n_frms = 1              # Just the frame number on the cmd line



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

cnt_q = 16*4*n_frms          # float32 words for quantiles
offs_qdat = 16*4*4*frm0      # Offset bytes for quantiles  

cnt_t = 16*n_frms            # float32 words for thresholds
cnt_r = cnt_t                # float32 words for residuals
offs_tdat = 16*4*frm0        # Bytes for thresholds
offs_rdat = offs_tdat        # Bytes for residuals

quantl = np.fromfile(fquantl_name, dtype=np.float32, \
                      offset=offs_qdat, count=cnt_q)
thresh = np.fromfile(fthresh_name, dtype=np.float32, \
                      offset=offs_tdat, count=cnt_t)
residl = np.fromfile(fresidl_name, dtype=np.float32, \
                      offset=offs_rdat, count=cnt_t)
if n_frms > 1:
    quantl = quantl.reshape((len(quantl)//64, 16, 4)) / 2500.
    thresh = thresh.reshape((len(thresh)//16, 16))
    residl = residl.reshape((len(residl)//16, 16))
else:
    quantl = quantl.reshape((16, 4)) / 2500.
    thresh = thresh.reshape((16))
    residl = residl.reshape((16))


F = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))  # Normal CDF

#
# Find 4 quantiles of the Normal PDF over 4 intervals separated by thresholds
# 
F_thr = F(-thresh) # Area under the Normal PDF curve over ]-Inf .. -thre]

if n_frms > 1:
    quanor = np.zeros((n_frms, 16, 4), dtype=np.float32)
    quanor[:,:,0] = quanor[:,:,3] = F_thr       # ]-Inf .. -thr]; [thr .. +Inf[ 
    quanor[:,:,1] = quanor[:,:,2] = 0.5 - F_thr # ]-thr .. 0];    ]0 .. +thr[ 
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

#
# Plot
#
F = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))  # Normal CDF

thr = 0.8

Fthr = F(-thr)
# hsnor = np.array([F(-thr), F(0)-F(-thr), F(thr) - F(0), 1 - F(0.92)])
hsnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])    # Normal quantilles

xrul = np.linspace(-3., 3., 51)
fnorm = 1/(2*np.pi)*np.exp(-xrul**2/2)

xloc = [-1.5, -0.5, 0.5, 1.5]

pl.figure()
pl.plot(xrul, 1.5*fnorm, 'b-.', lw=0.5)
pl.bar(xloc, hsnor, width=0.5) # , color='b') # , alpha=0.9)
pl.bar(xloc, quantl[0,:], width=0.2, color='orange')

    
pl.show()
