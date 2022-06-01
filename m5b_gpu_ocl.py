import os, sys
import numpy as np
import matplotlib.pyplot as pl
import pyopencl as cl
import time

tic = time.time()

fname = 'rd1910_wz_268-1811.m5b'

Nwitem_max = 256

nfrm = np.uint32(100)

argc = len(sys.argv)
if argc > 1:
    nfrm = np.uint32(sys.argv[1])

# print("argc = {0},  sys.argv = ".format(argc), sys.argv)
# print()


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

print()
print("argc = {0},  sys.argv = ".format(argc), sys.argv)
print()



# nfrm = np.uint64(100)  # sizeof(size_t) = CL_DEVICE_ADDRESS_BITS = 64 here */

# nfrm = total_frms; # Uncomment to read in the TOTAL M5B FILE
    
dat = np.fromfile(fname, dtype=np.uint32, count=frmwords*nfrm)

toc = time.time()

print("M5B file has been read. Time: %7.3f s.\n" % (toc-tic))

tic = time.time()

#
# ch_mask[16]:       Channel 2-bit masks
# quantl[nfrm,16,4]: Quantiles
# residl[nfrm,16]:   Residuals
# thresh[nfrm,16]:   Thresholds
# flag[nfrm,16]:     Flags
# niter[nfrm,16]:    Number of fminbnd() iterations 
#
ch_mask = np.zeros(16, dtype=np.uint32)           # Channel 2-bit masks
quantl = np.zeros((nfrm*16*4), dtype=np.float32)  # Quantiles
residl = np.zeros((nfrm*16), dtype=np.float32)    # Residuals
thresh = np.zeros((nfrm*16), dtype=np.float32)    # Thresholds
flag =   np.zeros((nfrm*16), dtype=np.uint16)     # Flags
niter =  np.zeros((nfrm*16), dtype=np.uint16)  # Number of iterations fminbnd()

#
# Find how many work groups/CUDA blocks and 
#                work items/CUDA threads per block needed
quot, rem = divmod(nfrm, Nwitem_max)
if quot == 0:
    Nwgroup = 1
    Nwitem = rem
elif rem == 0:
    Nwgroup = quot
    Nwitem = Nwitem_max
else:            # Both quot and rem != 0: last w-group will be < Nwitem_max 
    Nwgroup = quot + 1
    Nwitem = Nwitem_max

Nproc = Nwitem*Nwgroup

print("nfrm = {0}: Nwgroup = {1}, Nwitem = {2}, workitems in last group: {3}"
      .format(nfrm, Nwgroup, Nwitem, rem))

# raise SystemExit

#
# Create 16 2-bit masks for 16 channels in ch_mask
#
ch_mask[0] = 0x00000003;  # Mask for ch 0: 0b00000000000000000000000000000011
for ich in range(1,16):
    ch_mask[ich] = ch_mask[ich-1] << 2;

print("M5B 2-bit stream masks:")
for ich in range(16):
    print("ch_mask[{0:>2}] = 0x{1:>08x} = 0b{1:032b}".format(ich, ch_mask[ich]))


mf = cl.mem_flags

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

#
# Create input () and output ()
# buffers in the GPU memory. The mf.COPY_HOST_PTR flag forces copying from
# the host buffer, , to the device buffer (referred as buf_)
# in the GPU memory.
#
buf_ch_mask = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ch_mask)
buf_dat = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dat)
buf_quantl = cl.Buffer(ctx, mf.WRITE_ONLY, quantl.nbytes)
buf_residl = cl.Buffer(ctx, mf.WRITE_ONLY, residl.nbytes)
buf_thresh = cl.Buffer(ctx, mf.WRITE_ONLY, thresh.nbytes)
buf_flag = cl.Buffer(ctx,   mf.WRITE_ONLY, flag.nbytes)
buf_niter = cl.Buffer(ctx,  mf.WRITE_ONLY, niter.nbytes)


#
# Read the kernel code from file
#
with open ("ker_m5b_gauss_test.cl") as fh: ker = fh.read()

prg = cl.Program(ctx, ker).build(options=['-I .'])

completeEvent = prg.m5b_gauss_test(queue, (Nproc,), (Nwitem,),
                 buf_ch_mask, buf_dat,  buf_quantl, buf_residl, 
                 buf_thresh,  buf_flag, buf_niter,  nfrm)

events = [completeEvent]

cl.enqueue_copy(queue, quantl, buf_quantl, wait_for=events)
cl.enqueue_copy(queue, residl, buf_residl, wait_for=events)
cl.enqueue_copy(queue, thresh, buf_thresh, wait_for=events)
cl.enqueue_copy(queue, flag, buf_flag, wait_for=events)
cl.enqueue_copy(queue, niter, buf_niter, wait_for=events)



queue.flush()
queue.finish()

buf_ch_mask.release()
buf_dat.release()
buf_quantl.release()
buf_residl.release()
buf_thresh.release()
buf_flag.release()
buf_niter.release()

toc = time.time()

print("GPU time: %ld s." % toc-tic)
