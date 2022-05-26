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

print("M5B file has been read. Time: %7.3f s.\n" % (toc-tic))

tic = time.time()

ch_mask = np.zeros((nfrm,16), dtype=np.uint32)    # Channel 2-bit masks
quantl = np.zeros((nfrm,16,4), dtype=np.float32)  # Quantiles
residl = np.zeros((nfrm,16), dtype=np.float32)    # Residuals
thresh = np.zeros((nfrm,16), dtype=np.float32)    # Thresholds
flag =   np.zeros((nfrm,16), dtype=np.uint16)     # Flags
niter =  np.zeros((nfrm,16), dtype=np.uint16) # Number of iterations fminbnd()

 # /*
 #  * Create 16 2-bit masks for 16 channels
 #  */
 # ch_mask[0] = 0x00000003;       /* Mask for channel 0 */
 # for (ich=1; ich<16; ich++)
 #     ch_mask[ich] = ch_mask[ich-1] << 2;
    
 # /* for (ich=0; ich<16; ich++) */
 # /*     printf("ch_mask[%2d] = %08x = %032b\n", ich, ch_mask[ich]); */


# mf = cl.mem_flags

# ctx = cl.create_some_context()
# queue = cl.CommandQueue(ctx)

# #
# # Create input () and output ()
# # buffers in the GPU memory. The mf.COPY_HOST_PTR flag forces copying from
# # the host buffer, , to the device buffer (referred as buf_)
# # in the GPU memory.
# #
# buf_ch_mask = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ch_mask)
# buf_dat = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dat)
# buf_quantl = cl.Buffer(ctx, mf.WRITE_ONLY, hostbuf=quantl)
# buf_residl = cl.Buffer(ctx, mf.WRITE_ONLY, hostbuf=residl)
# buf_thresh = cl.Buffer(ctx, mf.WRITE_ONLY, hostbuf=thresh)
# buf_flag = cl.Buffer(ctx, mf.WRITE_ONLY, hostbuf=flag)
# buf_niter = cl.Buffer(ctx, mf.WRITE_ONLY, hostbuf=niter)


#
# Read the kernel code from file
#
# with open ("gauss_test_m5b.cl") as fh: ker = fh.read()

# prg = cl.Program(ctx, ker).build(options=['-I .'])

# prg.gausstestm5b(queue, (Nproc,), (Nwitem,), buf_ch_mask, buf_dat,
#                  buf_quantl, buf_residl, buf_thresh, buf_flag,
#                  buf_niter, nfrm)

# cl.enqueue_copy(queue, quantl, buf_quantl)
# cl.enqueue_copy(queue, residl, buf_residl)
# cl.enqueue_copy(queue, thresh, buf_thresh)
# cl.enqueue_copy(queue, flag, buf_flag)
# cl.enqueue_copy(queue, niter, buf_niter)



# queue.flush()
# queue.finish()

# buf_ch_mask.release()
# buf_dat.release()
# buf_quantl.release()
# buf_residl.release()
# buf_thresh.release()
# buf_flag.release()
# buf_niter.release()

# toc = time.time()

# print("GPU time: %ld s." % toc-tic)
