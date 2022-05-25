import os
import numpy as np
import matplotlib.pyplot as pl
import pyopencl as cl
import time

tic = time.time()

fname = 'rd1910_wz_268-1811.m5b'

Nwitem = 256
Nwgroup = 72
Nproc = Nwitem*Nwgroup


fmega = pow(1024.,2)
fgiga = pow(1024.,3)
frmwords = 2504;       # 32-bit words in one frame including the 4-word header
frmbytes = 2504*4
nfdat = 2500;          # 32-bit words of data in one frame


m5bbytes = os.path.getsize(fname)

total_frms = m5bbytes // frmbytes;
last_frmbytes = m5bbytes % frmbytes;
last_frmwords = last_frmbytes // 4;

print("File: rd1910_wz_268-1811.m5b")
print("M5B file size: %ld bytes = %g MiB = %g GiB" %
      (m5bbytes, m5bbytes/fmega, m5bbytes/fgiga))
print("Frame size: %d Bytes = %d words." % (frmbytes, nfdat))
print("Number of whole frames: %d" % total_frms)
print("Last frame size: %d Bytes = %d words." %
      (last_frmbytes, last_frmwords))

nfrm = 100

# nfrm = total_frms; # Uncomment to read in the TOTAL M5B FILE
    
dat = np.fromfile(fname, dtype=np.uint32, count=frmwords*nfrm)

toc = time.time()

print("M5B file has been read. Time: %d s.\n", toc - tic);

tic = time.time()

qua =   np.zeros((nfrm,16,4), dtype=np.float32)
qresd = np.zeros((nfrm,16), dtype=np.float32)
thr =   np.zeros((nfrm,16), dtype=np.float32)
flag =  np.zeros((nfrm,16), dtype=np.uint32)
niter = np.zeros((nfrm,16), dtype=np.uint32)

# mf = cl.mem_flags

# ctx = cl.create_some_context()
# queue = cl.CommandQueue(ctx)

# #
# # Create input () and output ()
# # buffers in the GPU memory. The mf.COPY_HOST_PTR flag forces copying from
# # the host buffer, , to the device buffer (referred as buf_)
# # in the GPU memory.
# #
# buf_rndst = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rndst)
# buf_rndu =  cl.Buffer(ctx, mf.WRITE_ONLY, rndu.nbytes)
buf_dat = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dat)




# queue.flush()
# queue.finish()
