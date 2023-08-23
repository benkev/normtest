help_text = \
'''
insert_code_in_m5b.py

'''
import numpy as np
import getopt, glob
import os, sys

n_bytes_uint32 = np.dtype(np.uint32).itemsize

#
# M5B file parameters
#
n_frmwords = 2504    # 32-bit words in one frame including the 4-word header
n_frmbytes = 2504*n_bytes_uint32  # Bytes in a frame with header
n_frmdatwords = 2500 # 32-bit words of data in one frame
n_frmdatbytes = 2500*n_bytes_uint32  # Bytes in a frame data section
    
#
# 
#
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
n_last_frm_bytes = n_m5bbytes % n_frmbytes
n_last_frm_words = n_last_frm_bytes // 4


n_codebytes = os.path.getsize(fcode)
sz_code = n_codewords = n_codebytes // 4
n_whole_codefrms = n_codebytes // n_frmdatbytes
n_whole_codefrm_words = n_frmdatwords*n_whole_codefrms
n_whole_codefrm_bytes = n_frmdatbytes*n_whole_codefrms
n_last_codefrm_bytes = n_codebytes % n_frmdatbytes
n_last_codefrm_words = n_last_codefrm_bytes // 4

#
# Read the whole code file into cod, type(cod) is raw bytes
#
with open(fcode, 'rb') as fc:
    cod = fc.read()


#
# Write the cod contents into the M5B file frames (not touching the frame
# headers) starting from frm0 frame
#
icod0 = 0
icod1 = n_frmdatbytes

m5b_offs = frm0*n_frmbytes + 4*n_bytes_uint32

with open(fm5b, 'r+b') as fm:
    for ifrm in range(n_whole_codefrms):
        fm.seek(m5b_offs)
        fm.write(cod[icod0:icod1])
        icod0 += n_frmdatbytes
        icod1 += n_frmdatbytes
        m5b_offs += n_frmbytes
      
    icod1 += n_last_codefrm_bytes
    fm.seek(m5b_offs)
    fm.write(cod[icod0:icod1])

    
