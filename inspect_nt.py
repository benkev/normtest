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
fchi2s = glob.glob("nt_chi2_*" + fm5b_base + "*" + ftst +".bin")

if len(fquantls) == 0 or len(fthreshs) == 0 or len(fchi2s) == 0:
    print("Files *%s*.bin with the timestamp \"%s\" not found." % \
          (fm5b_base, ftst))
    raise SystemExit

if len(fquantls) > 1 or len(fthreshs) > 1 or len(fchi2s) > 1:
    print("Ambiguous timestamp \"%s\". Give more detail." % ftst)
    raise SystemExit

#raise SystemExit

fquantl_name = fquantls[0]
fthresh_name = fthreshs[0]
fchi2_name = fchi2s[0]

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
chi2 = np.fromfile(fchi2_name, dtype=np.float32, \
                      offset=offs_rdat, count=cnt_t)

quantl = quantl.reshape(n_frms, 16, 4))
thresh = thresh.reshape((n_frms, 16))
chi2 = chi2.reshape((n_frms, 16))

#
# Average the observed frequencies over n_frms frames in each of 16 channels
#
q_obs = quantl.mean(axis=0) # q_obs.shape=(16,4)

F = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))  # Normal CDF

#
# Find 4 quantiles of the Normal PDF over 4 intervals separated by thresholds
# 
F_thr = F(-thresh) # Area under the Normal PDF curve over ]-Inf .. -thre]

# q_nor = np.zeros((n_frms, 16, 4), dtype=np.float32)
# q_nor[:,:,0] = q_nor[:,:,3] = F_thr       # ]-Inf .. -thr]; [thr .. +Inf[ 
# q_nor[:,:,1] = q_nor[:,:,2] = 0.5 - F_thr # ]-thr .. 0];    ]0 .. +thr[ 

# eps2 = np.zeros(16, dtype=np.float32)
    
# for ich in range(16):
#     eps2[ich] = np.sum((q_nor[ich,:] - quantl[ich,:])**2)


#
# Plot
#

Fthr = F(-thr)
q_nor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])    # Normal quantilles

xrul = np.linspace(-3., 3., 51)
fnorm = 1/(2*np.pi)*np.exp(-xrul**2/2)

xloc = [-1.5, -0.5, 0.5, 1.5]

pl.figure()
pl.plot(xrul, 1.5*fnorm, 'b-.', lw=0.5)
pl.bar(xloc, q_nor, width=0.5) # , color='b') # , alpha=0.9)
pl.bar(xloc, quantl[0,:], width=0.2, color='orange')

    
pl.show()
