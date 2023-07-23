help_text = \
'''
#
#   normtest_m5b_cuda.py
# 
# Normality (Gaussianity) test for M5B files on Nvidia GPU
# using PyCUDA package.
# Single precision floats.
#
# Requires:
# ker_m5b_gauss_test_chi2.cu, CUDA kernel.
#
# Usage: 
# $ python normtest_m5b_cuda.py <m5b-file-name> [<# of threads per block>] [-s]
#
# Or, from IPython:
#
# $ ipython --pylab
# %run normtest_m5b_cuda.py <m5b-file-name> [<# of threads per block>] [-s]
#
# If # of threads is not specified, the optimal (appearingly) 8 is used.
#
# If "-s" is present at the end of command line the results are saved in text
# files with names:
#    thresholds_*.txt
#    residuals_*.txt
#    n_iterations_*.txt
#    flags_*.txt
#    quantiles_*.txt
#
# Note that saving the results files may take long time, 
#
'''

import os, sys
import numpy as np
import matplotlib.pyplot as pl
import time
import pycuda.autoinit
# from pycuda import driver, compiler, gpuarray, tools
import pycuda as cu
import pycuda.gpuarray as gpuarray
import pycuda.compiler as compiler

def kernel_meminfo(kernel):
    shared=kernel.shared_size_bytes
    regs=kernel.num_regs
    local=kernel.local_size_bytes
    const=kernel.const_size_bytes
    # mbpt=kernel.max_threads_per_block
    print("Kernel memory: Local: %dB, Shared: %dB, Registers: %d, Const: %dB"
          % (local,shared,regs,const))

tic = time.time()

# fname = 'rd1910_wz_268-1811.m5b'

#
# It looks like 8 threads per block is the
# optimum: 2.245 s to do 1.2 Gb m5b file!
#
Nthreads_max = 8 #16 # 1024 # 32  # 256

saveResults = False   # Do not save the results in files by defauly
badSaveArg = False

print("sys.argv = ", sys.argv)

argc = len(sys.argv)
if argc == 1:
    print(help_text)
    raise SystemExit
if argc == 2:
    if sys.argv[1] in {'?', '-h', '--help'}:
        print(help_text)
        raise SystemExit
    else:
        fname = sys.argv[1]
if argc == 3:
    fname = sys.argv[1]
    Nthreads_max = int(sys.argv[2])
if argc == 4:
    fname = sys.argv[1]
    Nthreads_max = int(sys.argv[2])
    if sys.argv[3] == '-s':
        saveResults = True
    else:
        badSaveArg = True
if argc > 4 or badSaveArg:
    print('Wrong arguments.')
    print('Usage: python normtest_m5b_cuda.py <m5b-file-name> ' \
          '[<# of threads per block>] [-s]')
    raise SystemExit
    

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
if last_frmbytes != 0:
    print("Last incomplete frame size: %d Bytes = %d words." %
          (last_frmbytes, last_frmwords))

mem = os.popen('free -b').readlines()
mem = mem[1].split()
tot_ram = float(mem[1]) 
avl_ram = float(mem[6]) 

print()
print('CPU RAM: total %5.2f GB, available %5.2f GB' % (tot_ram/2**30,
                                                     avl_ram/2**30))
# nfrm = np.uint32(100)

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
# ch_mask = np.zeros(16, dtype=np.uint32)           # Channel 2-bit masks
# quantl = np.zeros((nfrm*16*4), dtype=np.float32)  # Quantiles
# residl = np.zeros((nfrm*16), dtype=np.float32)    # Residuals
# thresh = np.zeros((nfrm*16), dtype=np.float32)    # Thresholds
# flag =   np.zeros((nfrm*16), dtype=np.uint16)     # Flags
# niter =  np.zeros((nfrm*16), dtype=np.uint16) # Number of iterations fminbnd()

#
# Find how many work groups/CUDA blocks and 
#               work items/CUDA threads per block needed
#
quot, rem = divmod(nfrm, Nthreads_max)
if quot == 0:
    Nblocks = 1
    Nthreads = rem
elif rem == 0:
    Nblocks = quot
    Nthreads = Nthreads_max
else:            # Both quot and rem != 0: last w-group will be < Nthreads_max 
    Nblocks = quot + 1
    Nthreads = Nthreads_max

Nblocks =  int(Nblocks)
Nthreads = int(Nthreads)
   
# Nproc = Nthreads*Nblocks

# print("nfrm = {0}: Nwgroup = {1}, Nwitem = {2}, workitems in last group: {3}"
#       .format(nfrm, Nwgroup, Nwitem, rem))
print('GPU Parameters:')
print("Processing %d frames using %d CUDA blocks, %d threads in each." %
      (nfrm, Nblocks, Nthreads))
if rem != 0:
      print("The last, incomplete block has %d threads." % rem)

# raise SystemExit

#
# Create 16 2-bit masks for 16 channels in ch_mask
#
ch_mask = np.zeros(16, dtype=np.uint32)           # Channel 2-bit masks
ch_mask[0] = 0x00000003;  # Mask for ch 0: 0b00000000000000000000000000000011
for ich in range(1,16):
    ch_mask[ich] = ch_mask[ich-1] << 2;

# print("M5B 2-bit stream masks:")
# for ich in range(16):
#   print("ch_mask[{0:>2}] = 0x{1:>08x} = 0b{1:032b}".format(ich, ch_mask[ich]))

#
# Read the kernel code from file into the string "ker"
#
kernel = "ker_m5b_gauss_test.cu"

with open (kernel) as fh: kernel_code = fh.read()

print("CUDA kernel file '%s' is used\n" % kernel)

# -- initialize the device
#import pycuda.autoinit
cu.driver.init()

#
# GPU information
#
(free,total) = cu.driver.mem_get_info()
print("Global memory occupancy: %5.2f%% free of the total %5.2f GB" %
      (free*100/total, total/fgiga))

#
# Transfer host (CPU) memory to device (GPU) memory 
#
dat_gpu =     gpuarray.to_gpu(dat)
ch_mask_gpu = gpuarray.to_gpu(ch_mask)

#
# Create empty gpu arrays for the result
#
quantl_gpu = gpuarray.empty((nfrm*16*4,), np.float32)
residl_gpu = gpuarray.empty((nfrm*16,), np.float32)
thresh_gpu = gpuarray.empty((nfrm*16,), np.float32)
flag_gpu =   gpuarray.empty((nfrm*16,), np.uint16)
niter_gpu =  gpuarray.empty((nfrm*16,), np.uint16)

#
# Compile the kernel code 
#
mod = compiler.SourceModule(kernel_code,
                            options=['-I /home/benkev/Work/normtest/'])

#
# Get the kernel function from the compiled module
#
m5b_gauss_test = mod.get_function("m5b_gauss_test")

#
# Print the kernem memory information
#
kernel_meminfo(m5b_gauss_test)

#
# Call the kernel on the GPU card
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

print("\nGPU time: %.3f s." % (toc-tic))

quantl = quantl.reshape(nfrm,16,4)
residl = residl.reshape(nfrm,16)
thresh = thresh.reshape(nfrm,16)
flag =   flag.reshape(nfrm,16)
niter =  niter.reshape(nfrm,16)

#
# Release GPU memory allocated to the large arrays
#
del quantl_gpu
del residl_gpu
del thresh_gpu
del flag_gpu
del niter_gpu
del dat_gpu
del ch_mask_gpu

if not saveResults:
    raise SystemExit

#
# Save results
#
print('\nSaving results started ...')

tic = time.time()

basefn = fname.split('.')[0]
cnfrm = str(nfrm)
tstamp = str(np.round(tic % 1000, 3))
fntail = basefn + '_cuda_' + cnfrm + '_frames' + '_t' + tstamp + '.txt'

with open('result/thresholds_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%8g ' % thresh[ifrm,ich])
        fh.write('\n')

with open('result/residuals_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%12g ' % residl[ifrm,ich])
        fh.write('\n')

with open('n_iterations_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%2hu ' % niter[ifrm,ich])
        fh.write('\n')

with open('result/flags_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%1hu ' % flag[ifrm,ich])
        fh.write('\n')

with open('result/quantiles_' + fntail, 'w') as fh:
    for ifrm in range(nfrm):
        for ich in range(16):
            fh.write('%g %g %g %g    ' % tuple(quantl[ifrm,ich,:]))
        fh.write('\n')

toc = time.time()

print('Saving results in files: %.3f s.' % (toc-tic))

        
