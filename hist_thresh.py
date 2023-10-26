import sys, os
import numpy as np
import matplotlib.pyplot as pl
import getopt, glob

pl.ion()
pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

opts, args = getopt.gnu_getopt(sys.argv[1:], '')

ftst = None
if len(args) == 1:
    fm5b = args[0]  # Name of *.m5b file
    ftst = None     # No timestamp of *.bin files
elif len(args) == 2:
    fm5b = args[0]  # Name of *.m5b file
    ftst = args[1]  # Timestamp of *.bin files
else:
    print("Run: hist_thresh.py fname.m5b [timestamp]")
    raise SystemExit

fm5b_base = os.path.basename(fm5b)
fm5b_base = os.path.splitext(fm5b_base)[0]

if ftst:
    fhists = glob.glob("nt_hist_*" + fm5b_base + "*" + ftst +".bin")
    fthreshs = glob.glob("nt_thresh_*" + fm5b_base + "*" + ftst +".bin")
    fchi2s = glob.glob("nt_chi2_*" + fm5b_base + "*" + ftst +".bin")
else:
    fhists = glob.glob("nt_hist_*" + fm5b_base + "*.bin")
    fthreshs = glob.glob("nt_thresh_*" + fm5b_base + "*.bin")
    fchi2s = glob.glob("nt_chi2_*" + fm5b_base + "*.bin")
    
if len(fhists) == 0 or len(fthreshs) == 0 or len(fchi2s) == 0:
#if len(fthreshs) == 0 or len(fchi2s) == 0:
    print("Files *%s*.bin with the timestamp \"%s\" not found." % \
          (fm5b_base, ftst))
    raise SystemExit

if ftst and (len(fhists) > 1 or len(fthreshs) > 1 or len(fchi2s) > 1):
#if ftst and (len(fthreshs) > 1 or len(fchi2s) > 1):
    print("Ambiguous timestamp \"%s\". Give more detail." % ftst)
    raise SystemExit

#raise SystemExit

if ftst:
    fhist_name = fhists[0]
    fthresh_name = fthreshs[0]
    fchi2_name = fchi2s[0]
else:
    fhist_name = max(fhists, key=os.path.getctime)
    fthresh_name = max(fthreshs, key=os.path.getctime)
    fchi2_name = max(fchi2s, key=os.path.getctime)
    
    

t0 = np.fromfile(fthresh_name, dtype=np.float32)
thresh = t0.reshape((len(t0)//16, 16))

# r0 = np.fromfile(fchi2_name, dtype=np.float32)
# q0 = np.fromfile(fhist_name, dtype=np.float32)

# hist = q0.reshape((len(q0)//64, 16, 4)) / 2500.
# chi2 = r0.reshape((len(r0)//16, 16))

nfrm = len(t0)//16

pl.figure()
pl.hist(t0, 70)
pl.title("Quantization Threshold Estimates", fontsize=16)
pl.xlabel(r"$\theta$ = (quantization threshold)/$\sigma$", fontsize=14)
pl.grid(1)

vl = pl.ylim()
vd = vl[1] - vl[0]
hl = pl.xlim()
hd = hl[1] - hl[0]
xtpos = hl[0] + 0.55*hd

if fm5b == "rd1903_ft_100-0950.m5b":
    pl.text(hl[0]+0.01, 0.92*vd, "File: (%d frames)" % nfrm, fontsize=14)
    pl.text(hl[0]+0.01, 0.86*vd, fm5b, ha="left", fontsize=14)
else:
    pl.text(xtpos, 0.92*vd, "File: (%d frames)" % nfrm, ha="left", fontsize=14)
    pl.text(xtpos, 0.86*vd, fm5b, ha="left", fontsize=14)


pl.show()



