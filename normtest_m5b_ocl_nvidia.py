help_text = \
'''
#
#   normtest_m5b_ocl_nvidia.py
# 
# Normality (Gaussianity) test for M5B files on GPU
# using PyOpenCL package.
# Single precision floats.
#
# Requires:
# ker_m5b_gauss_test.cl, OpenCL kernel.
#
# Usage: 
# $ python normtest_m5b_cuda.py <m5b-file-name> [<# of threads per block>] [-s]
#
# Or, from IPython:
#
# $ ipython --pylab
# %run normtest_m5b_ocl_nvidia.py <m5b-file-name> [<# of threads per block>] \
#                                 [-s]
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
import pyopencl as cl
import time

tic = time.time()

# fname = 'rd1910_wz_268-1811.m5b'

#
# It looks like 8 work items (threads) per work group (block) is optimum:
#
wgsize = 8  # Work group size (# of work items in group)

argc = len(sys.argv)

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
    wgsize = int(sys.argv[2])
if argc == 4:
    fname = sys.argv[1]
    wgsize = int(sys.argv[2])
    if sys.argv[3] == '-s':
        saveResults = True
    else:
        badSaveArg = True
if argc > 4 or badSaveArg:
    print('Wrong arguments.')
    print('Usage: python normtest_m5b_cuda.py <m5b-file-name> ' \
          '[<# of threads per block>] [-s]')
    raise SystemExit
    

# raise SystemExit

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

plats = cl.get_platforms()
plat = plats[0]
devs = plat.get_devices(cl.device_type.GPU)
dev = devs[0]

print()
print('GPU: %s, %s' % (plat.name, plat.vendor))
print('Memory (global): %5.2f GB' % (dev.global_mem_size/2**30))
print('Memory (local):  %5.1f kB' % (dev.local_mem_size/2**10))
print('Max compute units %d' % (dev.max_compute_units))

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

print()
print("M5B file has been read. Time: %.3f s.\n" % (toc-tic))


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
# Find wiglobal >= nfrm, the total number of work items divisable by the
# local work size, wgsize (work group size)
#
wiglobal = int(wgsize*np.ceil(nfrm/wgsize))   # In kernel: get_global_size(0)
nwg = wiglobal//wgsize  # Number of work groups
rem = nfrm % wgsize

print('GPU Parameters:')
print("Processing %d frames using %d OpenCL work groups, " \
      "%d work items in each." % (nfrm, nwg, wgsize))
if rem != 0:
    print("The last, incomplete work group has %d work items." % (rem))

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
# Read the kernel code from file into the string "ker"
#
kernel = "ker_m5b_gauss_test.cl"

with open (kernel) as fh: ker = fh.read()

print("OpenCL kernel file '%s' is used\n" % kernel)

#
# GPU information
#
print('PyOpenCL version: ' + cl.VERSION_TEXT)
print('OpenCL header version: ' +
      '.'.join(map(str, cl.get_cl_header_version())) + '\n')
# Get installed platforms (SDKs)
print('- Installed platforms (SDKs) and available devices:')
platforms = cl.get_platforms()
plat = platforms[0]
# Get and print platform info
print('{} ({})'.format(plat.name, plat.vendor))
devices = plat.get_devices(cl.device_type.GPU)
dev = devices[0]
print(dev.name, dev.vendor)
print('Type: %s' % cl.device_type.to_string(dev.type))
print('Memory (global): %5.2 GB' % (dev.global_mem_size/2**30))
print('Memory (local): %5.1 kB' % (dev.local_mem_size/2**10))
print('Address bits: %d' % (dev.address_bits))
print('Max compute units %d' % (dev.max_compute_units))
print('Max work group size: %d' % (dev.max_work_group_size))
print('Max work item dims: %d' % (dev.max_work_item_dimensions))
print('Driver version: ', dev.driver_version)


prg = cl.Program(ctx, ker).build(options=['-I . -D __nvidia'])
#prg = cl.Program(ctx, ker).build(options=['-I /home/benkev/Work/normtest'])

# prg.m5b_gauss_test(queue, (nfrm,), None,
prg.m5b_gauss_test(queue, (wiglobal,), (wgsize,),
                 buf_dat, buf_ch_mask,  buf_quantl, buf_residl, 
                 buf_thresh,  buf_flag, buf_niter,  nfrm).wait()

cl.enqueue_copy(queue, quantl, buf_quantl)
cl.enqueue_copy(queue, residl, buf_residl)
cl.enqueue_copy(queue, thresh, buf_thresh)
cl.enqueue_copy(queue, flag, buf_flag)
cl.enqueue_copy(queue, niter, buf_niter)

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

print("GPU time: %.3f s." % (toc-tic))

quantl = quantl.reshape(nfrm,16,4)
residl = residl.reshape(nfrm,16)
thresh = thresh.reshape(nfrm,16)
flag = flag.reshape(nfrm,16)
niter = niter.reshape(nfrm,16)

raise SystemExit

#
# Save results
#

tic = time.time()

basefn = fname.split('.')[0]
cnfrm = str(nfrm)
tstamp = str(np.round(tic % 1000, 3))
fntail = basefn + '_ocl_' + cnfrm + '_frames' + '_t' + tstamp + '.txt'

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

with open('result/n_iterations_' + fntail, 'w') as fh:
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

        
