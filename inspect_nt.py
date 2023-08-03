#
# inspect_nt.py
#
# This script creates 4x4 plots of 16 histograms for each of the 16 channels.
# The plots are for one or several (averaged) frames. The histograms are
# compared with the normal distribution curves showing approximately from what
# size quantiles the observation data are drawn. For each plot, the chi^2 is
# printed, as well as the quantization threshold.
#
# The data are read from the *.bin files created with the gpu_m5b_chi2.py.
#
# Running:
# %run inspect_nt.py <m5b_filename> <timestamp> <start_frame_#> <#_of_frames>
# or
# python inspect_nt.py <m5b_filename> <nt_file_timestamp> <start_frame#> \
#          [<n_frames>]         
#


import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
import scipy.stats
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

quantl = quantl.reshape((n_frms, 16, 4))
thresh = thresh.reshape((n_frms, 16))
chi2 = chi2.reshape((n_frms, 16))

#
# Average the observed frequencies, thresholds, and chi2 over n_frms frames
# in each of the 16 channels
#
q_obs = quantl.mean(axis=0)  # q_obs.shape=(16,4)
th_obs = thresh.mean(axis=0) # th_obs.shape=(16,)
c2_obs = chi2.mean(axis=0)   # c2_obs.shape = (16,)


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
# alpha: level of significance
# df: number of degrees of freedom
#
alpha = 0.05    # level of significance
df = 3          # number of degrees of freedom

chi2cr = scipy.stats.chi2.ppf(1-alpha, df)

print('Chi^2 critical value at significance %.2f and df = %d: %.2f' % \
      (alpha, 3, chi2cr))


#
# Plot
#
pl.ion()  # Interactive mode; pl.ioff() - revert to non-interactive.
print("pl.isinteractive() -> ", pl.isinteractive())

# Fthr = F(-thr)
# q_nor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])    # Normal quantilles

# xrul = np.linspace(-3., 3., 51)
# fnorm = 1/(2*np.pi)*np.exp(-xrul**2/2)

# xloc = [-1.5, -0.5, 0.5, 1.5]

# pl.figure()
# pl.plot(xrul, 1.5*fnorm, 'b-.', lw=0.5)
# pl.bar(xloc, q_nor, width=0.5) # , color='b') # , alpha=0.9)
# pl.bar(xloc, quantl[0,:], width=0.2, color='orange')


fnorm = lambda x: (1/np.sqrt(2*np.pi))*(np.exp(-0.5*x**2))  # Normal PDF (0,1)
Fnorm = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))               # Normal CDF (0,1)

#hsep = 0.3
hsep = 0.2

# thr = 0.1*np.arange(1,17)
xrul = np.linspace(-3., 3., 1001)
xloc = np.array([-2, -0.5, 0.5, 1.5])
f_pdf = (1/np.sqrt(2*np.pi))*(np.exp(-0.5*xrul**2))  # Normal PDF (0,1)
f_pdf = (hsep-0.02)*f_pdf/f_pdf.max() # Make curve just under separation line

pl.figure(figsize=(8,8))
pl.figtext(0.5, 0.96, r"Observed \& Normal PDF Quantiles Separated by " \
           r"$-\infty, -\theta, \, 0, +\theta, +\infty$",
           fontsize=16, ha='center')

for ich in range(16):
    pl.subplot(4, 4, ich+1)

    thr = th_obs[ich]
    qua = 1.9*q_obs[ich,:]/2500

    Fthr = Fnorm(-thr)
    qnor = 1.9*np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])
    
    barloc = np.array([-2.5, -thr/2, thr/2, 2.5])
    pl.plot(xrul, f_pdf, 'orange')
    pl.fill_between(xrul, f_pdf, where=(-10<xrul)&(xrul<=-thr), color='b')
    pl.fill_between(xrul, f_pdf, where=(-thr<xrul)&(xrul<=0), color='m');
    pl.fill_between(xrul, f_pdf, where=(0<xrul)&(xrul<=thr), color='g')
    pl.fill_between(xrul, f_pdf, where=((thr<xrul)&(xrul<10)), color='c')

    xl = pl.xlim()
    pl.plot([xl[0],xl[1]], [hsep,hsep], 'k--', lw=1) # Horiz. separation line
    # pl.plot([-thr,-thr], [0,0.6], 'k--', lw=0.7)   # Vert. line -threshold
    # pl.plot([thr,thr], [0,0.6], 'k--', lw=0.7)     # Vert. line +threshold
    pl.plot([-thr,-thr], [0,0.75], 'k--', lw=0.7)   # Vert. line -threshold
    pl.plot([thr,thr], [0,0.75], 'k--', lw=0.7)     # Vert. line +threshold
    pl.xticks([-2,0,2], ["$-2\sigma$", "$0$", "$2\sigma$"], fontsize=9)
    pl.yticks([])
    # pl.text(-thr-0.7, 0.65, r"$-\theta$")
    # pl.text(thr+0.05, 0.65, r"$+\theta$")
    pl.text(-thr-0.7, 0.78, r"$-\theta$", fontsize=8)
    pl.text(thr+0.05, 0.78, r"$+\theta$", fontsize=8)
    
    bw = 0.15 if thr < 0.4 else 0.25  # Width of the histogram boxes
    pl.bar(barloc, qnor, bw*2, hsep+0.03, color='c')   # , edgecolor='k')
    pl.bar(barloc, qua, bw/2, hsep+0.03, color='k')
    pl.plot(barloc, qua+hsep+0.05, 'ro', markersize=1.5)
    pl.ylim(-0.03, 1.0)
    # pl.ylim(-0.05, 0.85)
    pl.text(xl[0], 0.9, r"$\theta=\pm%4.2f$;" % thr, fontsize=8)
    pl.text(-0.6, 0.9, r"$\chi^2=%4.1f$;" % c2_obs[ich], fontsize=8)
    pl.text(xl[1]-1.1, 0.9, "ch %2d" % ich, fontsize=8)
    #pl.grid(1)


# pl.figtext(0.5, 0.05, r"* Quantization Thresholds $\pm\theta$: " \
#            r"MUST be $|\theta|\, > 0.6745 \, \sigma$", \
#            fontsize=20, ha='center')
pl.figtext(0.5, 0.005, r"File: %s; Start frame %d, frames %d" %
           (fm5b, frm0, n_frms), fontsize=13, ha='center')

pl.subplots_adjust(top=0.935, bottom=0.055, left=0.035, right=0.965,
                   hspace=0.2, wspace=0.2)


    
pl.show()





