help_text = \
'''
print_frame_chan_chi2.py <chi2 file name .bin> <chi2_threshold value>

Print a table of frame #, channel #, and chi^2 for chi^2 > chi2_threshold

Sometimes a few channels in good data are flagged just because of random 
fluctuations in the data itself, and not because an alien piece of code was 
injected. Normally, in cases of fluctuations, chi^2 only slightly exceeds its
critical value chi^2_cr = 7.81. In contrast, a really foreign piece of code 
or data damages the whole frame or many frames in a row, and the chi^2 values
in such bad frames are many times larger than chi^2_cr. In order to spot such
frames we should roughen the Pearson's chi^2 criterion by searching for the
frames where, say, chi^2 > 20 or even 50.

Example. Print out the frames and channels where chi2 > 50:

%run print_frame_chan_chi2.py \
          nt_chi2_cuda_rd1910_wz_268-1811_bin_code_20231024_125630.067.bin 50.
 
'''

import sys
import numpy as np

if len(sys.argv) < 3:
    print(help_text)
    sys.exit(0)

fname_chi2_bin = sys.argv[1]
chi2_threshold = float(sys.argv[2])

c2 = np.fromfile(fname_chi2_bin, dtype=np.float32) # Read chi^2 into 1D array c2
c2 = c2.reshape((len(c2)//16,16))     # Convert it into 2D table c2[frame,chan]

ic = np.where(c2 > chi2_threshold) # Find indices into c2

#
# Print a table of frame #, channel #, and chi^2 > 50:
#
for i in range(len(ic[0])):
    print(ic[0][i], ic[1][i], c2[ic[0][i],ic[1][i]])
