#
#   m5b_cuda.py
# 
# Normality (Gaussianity) test for M5B files on GPU
# Single precision floats.
#
# Requires:
# ker_m5b_gauss_test.cl, OpenCL kernel.
#


import os, sys
import numpy as np
import matplotlib.pyplot as pl
import time
from pycuda import driver, compiler, gpuarray, tools

tic = time.time()

fname = 'rd1910_wz_268-1811.m5b'

#
# It looks like 8 threads per block is optimum: 2.245 s to do 1.2 Gb m5b file!
#
Nwitem_max = 8 #16 # 1024 # 32  # 256

if len(sys.argv) > 1:
    Nwitem_max = int(sys.argv[1])

nfrm = np.uint32(100)

argc = len(sys.argv)
if argc > 1:
    nfrm = np.uint32(sys.argv[1])
if argc > 2:
    nwg = np.uint32(sys.argv[2])

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

# sizeof(size_t) = CL_DEVICE_ADDRESS_BITS = 64 here

nfrm = np.uint32(total_frms); # Uncomment to read in the TOTAL M5B FILE
    
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
# residl = np.zeros((nfrm*16), dtype=np.float32)    # Residuals
# thresh = np.zeros((nfrm*16), dtype=np.float32)    # Thresholds
# flag =   np.zeros((nfrm*16), dtype=np.uint16)     # Flags
# niter =  np.zeros((nfrm*16), dtype=np.uint16) # Number of iterations fminbnd()

#
# Find how many work groups/CUDA blocks and 
#               work items/CUDA threads per block needed
#
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

Nblocks =  int(Nwgroup)
Nthreads = int(Nwitem)
    
# Nproc = Nwitem*Nwgroup

print("nfrm = {0}: Nwgroup = {1}, Nwitem = {2}, workitems in last group: {3}"
      .format(nfrm, Nwgroup, Nwitem, rem))
print("nfrm = {0}: Nblocks = {1}, Nthreads = {2}, threads in last group: {3}"
      .format(nfrm, Nblocks, Nthreads, rem))

# raise SystemExit

#
# Create 16 2-bit masks for 16 channels in ch_mask
#
ch_mask[0] = 0x00000003;  # Mask for ch 0: 0b00000000000000000000000000000011
for ich in range(1,16):
    ch_mask[ich] = ch_mask[ich-1] << 2;

# print("M5B 2-bit stream masks:")
# for ich in range(16):
#   print("ch_mask[{0:>2}] = 0x{1:>08x} = 0b{1:032b}".format(ich, ch_mask[ich]))

#
# Read the kernel code from file into the string "ker"
#
with open ("ker_m5b_gauss_test.cu") as fh: kernel_code = fh.read()

# -- initialize the device
import pycuda.autoinit

#
# Transfer host (CPU) memory to device (GPU) memory 
#
dat_gpu =     gpuarray.to_gpu(dat)
ch_mask_gpu = gpuarray.to_gpu(ch_mask)

#
# Create empty gpu array for the result (C = A * B)
#
quantl_gpu = gpuarray.empty((nfrm*16*4,), np.float32)
residl_gpu = gpuarray.empty((nfrm*16,), np.float32)
thresh_gpu = gpuarray.empty((nfrm*16,), np.float32)
flag_gpu = gpuarray.empty((nfrm*16,), np.uint16)
niter_gpu = gpuarray.empty((nfrm*16,), np.uint16)

#
# Compile the kernel code 
#
mod = compiler.SourceModule(kernel_code,
#                            options=['-I .'])
#                            options=['-I ~/Work/normtest'])
                            options=['-I /home/benkev/Work/normtest/'])

#
# Get the kernel function from the compiled module
#
m5b_gauss_test = mod.get_function("m5b_gauss_test")

#
# Call the kernel on the card
#
m5b_gauss_test(dat_gpu, ch_mask_gpu,  quantl_gpu, residl_gpu, 
               thresh_gpu,  flag_gpu, niter_gpu,  nfrm,
               block = (Nthreads, 1, 1), grid = (Nblocks, 1))
#               block = (int(nfrm), 1, 1))
#               block=(nthreads,1,1), grid=(nblk,1)


quantl = quantl_gpu.get()
residl = residl_gpu.get()
thresh = thresh_gpu.get()
flag =   flag_gpu.get()
niter =  niter_gpu.get()

toc = time.time()

print("GPU time: %.3f s." % (toc-tic))

quantl = quantl.reshape(nfrm,16,4)
residl = residl.reshape(nfrm,16)
thresh = thresh.reshape(nfrm,16)
flag =   flag.reshape(nfrm,16)
niter =  niter.reshape(nfrm,16)

# raise SystemExit

#
# Save results
#

tic = time.time()

basefn = fname.split('.')[0]
cnfrm = str(nfrm)

fntail = basefn + '_cuda_' + cnfrm + '_frames.txt'

with open('thresholds_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%8g ' % thresh[ifrm,ich])
        fh.write('\n')

with open('residuals_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%12g ' % residl[ifrm,ich])
        fh.write('\n')

with open('n_iterations_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%2hu ' % niter[ifrm,ich])
        fh.write('\n')

with open('flags_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%1hu ' % flag[ifrm,ich])
        fh.write('\n')

with open('quantiles_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%g %g %g %g    ' % tuple(quantl[ifrm,ich,:]))
        fh.write('\n')

toc = time.time()

print('Saving results in files: %.3f s.' % (toc-tic))

        
