help_text = \
'''


'''
import numpy as np
import getopt, glob
import os, sys

n_bytes_uint32 = np.dtype(np.uint32).itemsize

#
# M5B file parameters
#
n_frmwords = 2504    # 32-bit words in one frame including the 4-word header
n_frmbytes = 2504*n_bytes_uint32  # Bytes in a frame
n_frmdatwords = 2500 # 32-bit words of data in one frame
    

opts, args = getopt.gnu_getopt(sys.argv[1:], '')

fm5b = args[0]   # M5B (or M5A) file name
fcode = args[1]  # Any file with (binary executable) code
frm0 = int(args[2])  # Start frame number to write
n_frms = 1

#
# Accounting
#
n_m5bbytes = os.path.getsize(fm5b)
sz_m5b = n_m5bwords = n_m5bbytes // 4
n_whole_frms = n_m5bbytes // n_frmbytes
n_whole_frm_words = n_frmwords*n_whole_frms
n_whole_frm_bytes = n_frmbytes*n_whole_frms
n_last_frmbytes = n_m5bbytes % n_frmbytes
n_last_frmwords = n_last_frmbytes // 4




