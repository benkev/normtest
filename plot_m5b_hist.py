help_text = \
'''
plot_m5b_hist.py <M5B file name> <timestamp>

Plots two histograms of the results from gpu_m5b.py for the whole 
M5B (or M5A) file: 
1. Distribution of chi^2 and a red marker showing position of the critical
     chi^2 value (7.81), as well as the percent of chi^2 exceeding it.
2. Distribution of the optimal quantization thresholds and a red marker
     showing position of the critical threshold value (0.6745 rms), as well as
     the percent of the thresholds that failed to reach it.

Example:
    %run plot_m5b_hist.py rd1910_wz_268-1811.m5b 793
'''

import sys, os
import numpy as np
import scipy.stats
import matplotlib.pyplot as pl
import getopt, glob

pl.ion()  # Interactive mode; pl.ioff() - revert to non-interactive.
pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

opts, args = getopt.gnu_getopt(sys.argv[1:], '')

if len(args) < 2:
    print(help_text)
    sys.exit(0)

fm5b = args[0]
ftst = args[1]

if not os.path.exists(fm5b):
    fm5b_list = glob.glob(fm5b + ".*")
    if len(fm5b_list) > 0:
        fm5b = fm5b_list[0]
fm5b_size = os.stat(fm5b).st_size/2**30 # Size in GiB
fm5b_size = ";   size %.2f GiB" % fm5b_size

fm5b_base = os.path.basename(fm5b)
fm5b_base = os.path.splitext(fm5b_base)[0]

fhists = glob.glob("nt_hist_*" + fm5b_base + "*" + ftst +".bin")
fthreshs = glob.glob("nt_thresh_*" + fm5b_base + "*" + ftst +".bin")
fchi2s = glob.glob("nt_chi2_*" + fm5b_base + "*" + ftst +".bin")

if len(fhists) == 0 or len(fthreshs) == 0 or len(fchi2s) == 0:
    print("Files *%s*.bin with the timestamp \"%s\" not found." % \
          (fm5b_base, ftst))
    raise SystemExit

if len(fhists) > 1 or len(fthreshs) > 1 or len(fchi2s) > 1:
    print("Ambiguous timestamp \"%s\". Give more detail." % ftst)
    raise SystemExit

fhist_name = fhists[0]
fthresh_name = fthreshs[0]
fchi2_name = fchi2s[0]


c2 = np.fromfile(fchi2_name, dtype=np.float32)
th = np.fromfile(fthresh_name, dtype=np.float32)
qu = np.fromfile(fhist_name, dtype=np.float32)

#
# Chi^2 critical value at significance
#
alpha = 0.05    # level of significance
df = 3          # number of degrees of freedom
chi2cr = scipy.stats.chi2.ppf(1-alpha, df)
print('Chi2 critical value at significance %.2f and df = %d: %.2f' % \
      (alpha, df, chi2cr))
c2_overcr = 100*len(np.where(c2 > chi2cr)[0])/c2.shape[0]
th_cr = 0.6745  # The critical value of quantization threshold
th_undercr = 100*len(np.where(th < th_cr)[0])/th.shape[0]
print(r"The critical value of quantization threshold: %.4f rms" % th_cr)
# print(r"The critical value of quantization threshold: "
#       "MUST be $\abs{\theta/\sigma} >$ %.4f" % th_cr)


pl.figure(figsize=(6,7))
pl.figtext(0.5, 0.95, "File "+fm5b+fm5b_size, fontsize=16, ha='center')

pl.subplot(2,1,1)
pl.grid(1)
pl.hist(c2, 100, label=r"$\chi^2$")
yl = pl.ylim()
pl.plot((chi2cr, chi2cr), (0, 0.7*yl[1]), 'r')
#xl = pl.xlim(); xr = xl[1] - xl[0];
pl.text(chi2cr, 0.85*yl[1], r"$\chi^2_{cr} =$ %.2f; " % chi2cr, fontsize=12,
        color='r')
pl.text(chi2cr, 0.75*yl[1], r"$\#>\chi^2_{cr}$: %.2f $\%%$" % c2_overcr,
        fontsize=12, color='k')
# pl.xlabel(r"$\chi^2$")
pl.legend()

pl.subplot(2,1,2)
pl.grid(1)
pl.hist(th, 100, label=r"$\theta$")
xl = pl.xlim(); xr = xl[1] - xl[0]; 
yl = pl.ylim()
pl.plot((th_cr, th_cr), (0, 0.7*yl[1]), 'r', lw=2)
xl = pl.xlim(); xr = xl[1] - xl[0]; 
pl.text(xl[0]+0.04*xr, 0.85*yl[1], r"$\theta_{cr} =$ %.2f; " % th_cr,
        fontsize=12, color='r')
pl.text(xl[0]+0.04*xr, 0.75*yl[1], r"$\#<\theta_{cr}$: %.1f $\%%$" %
        th_undercr, fontsize=12, color='k')
# pl.xlabel(r"$\theta$")
pl.legend()

pl.subplots_adjust(top=0.91, bottom=0.07, left=0.125, right=0.95, hspace=0.2,
                   wspace=0.2)

pl.show()

