help_text = \
'''
gpu_m5b.py:

Normality (Gaussianity) test for M5B files on either an Nvidia GPU using PyCUDA
package or a GPU using PyOpenCL package. Single precision floats.

This module provides "transparent" access to the GPU independent of the
software framework used, CUDA or OpenCL.

The module contains class Normtest. It is not intended to create multiple
class instances (althoug it is surely possible). When imported, it 
probes the system to find what GPU frameworks are installed. It chooses
automatically between PyCUDA and OpenCL and initializes the relevant 
data structures. In case both frameworks are installed, it prefers
PyCUDA since it is ~1.5 times faster than PyOpenCL. 

--------- FUTURE WORK --------------------------- FUTURE WORK ----------------
However, it is possible to overwrite the framework using the class method

Normtest.set_fw(fw="").

Parameter:
    fw: framework to use. It can be "cuda", "opencl", or None.
        If fw="", nothing changes. 
--------- FUTURE WORK --------------------------- FUTURE WORK ----------------

    Class method Normtest.do_m5b(m5b_filename [, nthreads=8])

The normtest class provides a "class method" do_m5b(m5b_filename),
which runs the normality test on the available GPU and the software
framework selected. If the M5B file is large and it does not fit into either
system RAM or the GPU ram, it is processed in chunks. The results are saved 
in binary files.
Parameter:
    nthreads: number of threads per block (in CUDA terminology), and, 
              which is the same, number of work items per work group 
              (in OpenCL terms).

The do_m5b(m5b_filename) method uploads the M5B file chunk into the GPU memory
and launches one of the kernels, ker_m5b_gauss_test.cu or 
ker_m5b_gauss_test.cl, dependent on the framework type. The results are saved
in the binary files. The file have the following names:

    nt_<data>_<framework>_<m5b_basename>_<timestamp>.bin,

where <data> is the result types:

quantl: dtype=np.float32, shape=(n_frames,16,4), 4 quantiles for 16 channels;
chi2:   dtype=np.float32, shape=(n_frames,16), chi^2 for 16 channels;
thresh: dtype=np.float32, shape=(n_frames,16), quantization thresholds found
        for 16 channels;
flag:   dtype=np.uint16, shape=(n_frames,16), flags for 16 channels; 
niter:  dtype=np.uint16, shape=(n_frames,16), number of iterations of Brent's
        optimization method used to find the optimal quantization threshold
        for 16 channels;

An example of the result file name: 
    nt_hist_cuda_rd1903_ft_100-0950_20230726_225857.829.bin

   
Usage example:
   
from gpu_m5b import Normtest as nt
nt.do_m5b("rd1910_wz_268-1811.m5b")

Empirically, it has been found that the best performance is achieved 
with 8 threads per block (in CUDA terminology), and, which is the same, 
8 work items per work group (in OpenCL terms). However, this number can 
be changed using the nthreads parameter. For example:

nt.do_m5b("rd1910_wz_268-1811.m5b", nthreads=64)

'''

import numpy as np
import importlib
import os, sys
import time
import warnings

class Normtest:

    name = 'Normtest'
    fmega = pow(1024.,2)
    fgiga = pow(1024.,3)

    n_bytes_uint32 = np.dtype(np.uint32).itemsize

    #
    # M5B file parameters
    #
    n_frmwords = 2504    # 32-bit words in one frame including the 4-word header
    n_frmbytes = 2504*n_bytes_uint32  # Bytes in a frame
    n_frmdatwords = 2500 # 32-bit words of data in one frame

    #quota_dat = 0.95  # Quota of dat array in overall GPU data (approx)
    #quota_dat = 0.90  # Quota of dat array in overall GPU data (approx)
    #quota_dat = 0.85  # Quota of dat array in overall GPU data (approx)
    quota_dat = 0.93    # Works on Nvidia GPUs
    quota_dat = 0.80    # ??????????????
    
    #
    # Determine if PyCUDA or/and PyOpenCL are installed
    #
    pycuda_installed = importlib.util.find_spec("pycuda") is not None
    pyopencl_installed = importlib.util.find_spec("pyopencl") is not None

    #
    # In case both are installed, prefer PyCUDA since it is ~1.5 times faster
    #    
    gpu_framework = ""
    ############################################################
    #                  USE OpenCL ONLY                         #
    ############################################################
    # if pycuda_installed:
    #     gpu_framework = "cuda"     # Use PyCUDA even if PyOpenCL installed
    # elif pyopencl_installed:
    #     gpu_framework = "opencl"   # Use PyOpenCL if PyCUDA not installed
    # else:
    #     warnings.filterwarnings("ignore")
    #     sys.exit("Neither PyCUDA nor PyOpenCL are installed. Exiting.")

    if pyopencl_installed:
        gpu_framework = "opencl"   # Use PyOpenCL
    else:
        warnings.filterwarnings("ignore")
        sys.exit("PyOpenCL not installed. Exiting.")

    import psutil
    mem = psutil.virtual_memory() 
    mem_cpu_total = mem.total
    mem_cpu_available = mem.available
    mem_cpu_used = mem.used
    mem_cpu = mem_cpu_total  # Use total of CPU memory
    sz_cpu = mem_cpu_total   # Use total of CPU memory
    
    #
    # Create 16 2-bit masks for 16 channels in ch_mask
    #
    ch_mask = np.zeros(16, dtype=np.uint32)           # Channel 2-bit masks
    ch_mask[0] = 0x00000003;  # For ch 0: 0b00000000000000000000000000000011
    for ich in range(1,16):
        ch_mask[ich] = ch_mask[ich-1] << 2;




    if gpu_framework == "cuda":
        import pycuda as cu
        import pycuda.autoinit
        import pycuda.compiler as compiler
        import pycuda.gpuarray as gpuarray
        
        cu_dev = cu.driver.Device(0)
        # dev_name = cu_dev.name()
        # mem_gpu_total = dev.total_memory()   # /2**30 GB
        
        (mem_gpu_free, mem_gpu_total) = cu.driver.mem_get_info()
        # mem_gpu = mem_gpu_free # Use all the available GPU global memory
        sz_gpu = mem_gpu_free  # Only use the available GPU global memory
        #sz_gpu = mem_gpu_total   # Use all the available GPU global memory
        print("GPU RAM size: %.2f GiB;    CPU RAM size: %.2f GiB" % \
              (sz_gpu/fgiga, sz_cpu/fgiga))

        #
        # Read the kernel code from file into the string "ker"
        #
        ker_filename = "ker_m5b_gauss_test.cu"
        with open (ker_filename) as fh: ker_source_code = fh.read()
        print("CUDA kernel file '%s' is used\n" % ker_filename)

        #
        # Compile the kernel code 
        #
        prog_cuda = compiler.SourceModule(ker_source_code,
                            options=['-I /home/benkev/Work/normtest/'])
        #
        # Get the kernel function from the compiled module
        #
        m5b_gauss_test_cuda = prog_cuda.get_function("m5b_gauss_test")



    if gpu_framework == "opencl":
        import pyopencl as cl

        #
        # Determine which OpenCL platform is installed: NVIDIA or AMD,
        # and set appropriate kernel compilation options (ker_opts)
        #
        plats = cl.get_platforms()
        plat = plats[0]
        platform = plat.name
        platform = platform.split()[0] # Get the first word

        if platform == 'NVIDIA':
        #    ker_opts = ['-I . -D __nvidia']
            ker_opts = ['-I . -D __amd']
        elif platform == 'AMD':
            ker_opts = ['-I . -D __amd']
        else:
            sys.exit("Platform ""%s"" is not supported. Exiting." % platform)

        devs = plat.get_devices(cl.device_type.ALL)
        ocl_dev = devs[0]
        #
        # No idea how to get _available_ global memory in OpenCL !!!!!!!!!!!!!!
        #
        mem_gpu_total = ocl_dev.global_mem_size 
        sz_gpu = mem_gpu_total   # Use all the available GPU global memory
        # sz_gpu = mem_gpu_free  # Only use the available GPU global memory ???
        print("GPU RAM size: %.2f GiB;    CPU RAM size: %.2f GiB" % \
              (sz_gpu/fgiga, sz_cpu/fgiga))

        # mf = cl.mem_flags
        ctx = cl.create_some_context()
        #
        # Read the kernel code from file into the string "ker"
        #
        ker_filename = "ker_m5b_gauss_test.cl"
        with open (ker_filename) as fh: ker_source_code = fh.read()
        print("OpenCL kernel file '%s' is used\n" % ker_filename)
        #
        # Compile the kernel code 
        #
        prog_opencl = cl.Program(ctx, ker_source_code).build(options=ker_opts)
        m5b_gauss_test_ocl = prog_opencl.m5b_gauss_test


    @classmethod
    def set_fw(cls, fw=""): # Set GPU framework to use, "cuda" or "opencl"
        pass

        
    @classmethod
    def do_m5b(cls, fname_m5b, nthreads=8):

        cls.fname_m5b = fname_m5b
        cls.n_threads_max = nthreads
        #
        # It looks like 8 threads per block is
        # the optimum: 2.245 s to do 1.2 Gb m5b file!
        #

        #
        # Accounting
        #
        n_m5bbytes = os.path.getsize(fname_m5b)
        n_m5bwords = n_m5bbytes // 4
        cls.sz_m5b = n_m5bbytes
        n_whole_frms = n_m5bbytes // cls.n_frmbytes
        n_whole_frm_words = cls.n_frmwords*n_whole_frms
        n_whole_frm_bytes = cls.n_frmbytes*n_whole_frms
        n_last_frmbytes = n_m5bbytes % cls.n_frmbytes
        n_last_frmwords = n_last_frmbytes // 4
        #
        # Select processing mode dependent on relation sz_m5b ~ sz_cpu ~ sz_gpu
        
        # If f == sz_m5b  (M5B file size),
        #    c == sz_cpu  (CPU memory size),
        #    g == sz_gpu  (GPU memory size),
        # then there are 6 = 3! options:
        #   f < c < g
        #   f < g < c
        #   c < f < g
        #   g < f < c
        #   g < c < f
        #   c < g < f        
        #

        f = sz_m5b = cls.sz_m5b
        c = sz_cpu = cls.sz_cpu
        g = sz_gpu = cls.sz_gpu

        fitsBoth = (f < c) and (f < g) # M5B file fits both CPU and GPU
        fitsNone = (f > c) and (f > g) # M5B file fits neither CPU nor GPU
        fitsCPUnotGPU = g < f < c      # M5B file does not fit CPU but fits GPU
        fitsGPUnotCPU = c < f < g      # M5B file does not fit GPU but fits CPU

        cls.fitsBoth = fitsBoth
        cls.fitsNone = fitsNone
        cls.fitsGPUnotCPU = fitsGPUnotCPU
        # cls.fitsGPUnotCPU = fitsGPUnotCPU   # WE DO NOT USE IT
        cls.fitsCPUnotGPU = fitsCPUnotGPU

        #
        # Determine the M5B file partitioning: how many frames it contains,
        # how many whole frames. How many whole frames fit a chunk size to read
        # into the dat array and process in one go. What is the size of the
        # last, incomplete chunk (if there is one). What are the sizes of
        # chunks and what are the read offsetts sizes in uint32 words.
        #
        # Assume the file chunk to read into dat array can be ~95% of available
        # GPU memory (quota_dat = .95). The rest, ~5%, will contain the work
        # variables and the result arrays.
        #
        # Maximum dat array size (bytes) to fit GPU along with other arrays 
        sz_dat_max_rough = int(cls.quota_dat*sz_gpu)
        # Maximum number of the whole frames in a chunk to read at once
        # to fit GPU memory
        n_frms_chunk_max = sz_dat_max_rough // cls.n_frmbytes
        # Maximum  size (bytes) of dat array made of whole frames to fit GPU
        # along with other arrays
        sz_dat_max = n_frms_chunk_max*cls.n_frmbytes

        #
        # If the actual file size less than the maximum allowed, the whole
        # file fits the memory, and the dat array size should be put to the
        # file size in whole frames. The whole file will be read at once.
        # Otherwise the file does not fit the memory and will be read by
        # chunks of the maximum allowed length.
        #
        if n_whole_frms > n_frms_chunk_max: # The m5b file does not fit memory
            sz_dat = sz_dat_max
            n_frms_chunk = n_frms_chunk_max
        else:                               # The m5b file fits memory
            sz_dat = n_whole_frm_bytes
            n_frms_chunk = n_whole_frms

        
        # Size of the file chunk of whole frames to read at once in 32bit words
        n_words_whole_chunk = n_frms_chunk*cls.n_frmwords
        n_bytes_whole_chunk = n_frms_chunk*cls.n_frmbytes
        # Number of whole chunks (each having n_frms_chunk of whole frames) in
        # the entire m5b file
        n_m5b_whole_chunks = n_m5bwords // n_words_whole_chunk
        # Number of uint32 words in the last, incomplete chunk of the m5b file.
        # The last, incomplete chunk may have the last frame incomplete
        n_words_last_chunk_incompl_frm = n_m5bwords % n_words_whole_chunk
        # Determine how many whole frames the last chunk has
        n_frms_last_chunk = n_words_last_chunk_incompl_frm // cls.n_frmwords
        # Number of words in the last (possibly incomplete) chunk to make
        # a whole number of frames
        n_words_last_chunk = n_frms_last_chunk*cls.n_frmwords
        n_bytes_last_chunk = n_frms_last_chunk*cls.n_frmbytes
        # If the last frame is incomplete, this will be non-zero
        n_words_last_incompl_frm = n_words_last_chunk_incompl_frm - \
                                   n_words_last_chunk

        #
        # Calculate number of m5b file chunks allowing for a possible
        # last incomplete chunk
        #
        n_m5b_chunks = n_m5b_whole_chunks  # Assume all chunks are whole 
        if n_words_last_chunk != 0:
            n_m5b_chunks = n_m5b_whole_chunks + 1 # Include the incomplete chunk

        cls.n_m5b_chunks = n_m5b_chunks
        cls.n_frms_chunk = n_frms_chunk
        cls.n_frms_last_chunk = n_frms_last_chunk
        cls.n_words_whole_chunk = n_words_whole_chunk
        cls.n_bytes_whole_chunk = n_bytes_whole_chunk
        cls.n_words_last_chunk = n_words_last_chunk
        cls.n_bytes_last_chunk = n_bytes_last_chunk
        
        # Number of frames in a whole file chunk and in dat array
        # n_frms_whole = np.uint32(n_words_whole_chunk // cls.n_frmwords)

        print("n_m5bbytes = ", n_m5bbytes, ", n_m5bwords = ", n_m5bwords)
        print("n_whole_frms = ", n_whole_frms)
        print("n_whole_frm_bytes = ", n_whole_frm_bytes)
        print("n_whole_frm_words = ", n_whole_frm_words)
        print("n_last_frmbytes = ", n_last_frmbytes)
        print("n_last_frmwords = ", n_last_frmwords)
        print("sz_gpu = ", sz_gpu)
        print("sz_dat_max_rough = ", sz_dat_max_rough)
        print("sz_dat_max = ", sz_dat_max)
        print("sz_dat = ", sz_dat)
        print("n_words_whole_chunk = ", n_words_whole_chunk)
        print("n_bytes_whole_chunk = ", n_bytes_whole_chunk)
        print("n_words_last_chunk = ", n_words_last_chunk)
        print("n_bytes_last_chunk = ", n_bytes_last_chunk)
        print("n_frms_chunk_max = ", n_frms_chunk_max)
        print("n_frms_chunk = ", n_frms_chunk)
        print("n_frms_last_chunk = ", n_frms_last_chunk)
        print("n_m5b_whole_chunks = ", n_m5b_whole_chunks)
        print("n_m5b_chunks = ", n_m5b_chunks)
        print("n_words_last_chunk_incompl_frm = ",
              n_words_last_chunk_incompl_frm)
        print("n_words_last_incompl_frm = ", n_words_last_incompl_frm)
        # print("chunk_size_words = ", chunk_size_words)
        # print("chunk_offs_words = ", chunk_offs_words)
        # print("n_frms_whole = ", n_frms_whole)
        print(" = ", )
        print(" = ", )
        print(" = ", )
        print(" = ", )
        print(" = ", )
        
        if cls.gpu_framework == "cuda":
            cls.do_m5b_cuda()
        elif cls.gpu_framework == "opencl":
            cls.do_m5b_opencl()


        

    @staticmethod    
    def form_fout_name(fname_m5b):
        '''
        Find components of the m5b file name to form the basename of
        the output files with results and the time stamp
        '''
        # fname_full = os.path.expanduser(fname_m5b)
        basefn_m5b = os.path.basename(fname_m5b) # Like "rd1910_wz_268-1811.m5b"
        basefn = os.path.splitext(basefn_m5b)[0] # Like "rd1910_wz_268-1811"
        # basefn = "nt_" + basefn                # Like "nt_rd1910_wz_268-1811"

        t_stamp = str(time.strftime("%Y%m%d_%H%M%S")) + \
            ".%03d" % (1000*np.modf(time.time())[0])

        # print("t_stamp = ", t_stamp)

        return basefn, t_stamp




    @classmethod
    def do_m5b_cuda(cls):

        ticg = time.time()

        n_threads_max = cls.n_threads_max

        #
        # Create empty gpu arrays for the results
        #
        n_frms_chunk = cls.n_frms_chunk
        
        gpuarray = cls.gpuarray
        quantl_gpu = gpuarray.empty((n_frms_chunk*16*4,), np.float32)
        chi2_gpu = gpuarray.empty((n_frms_chunk*16,), np.float32)
        thresh_gpu = gpuarray.empty((n_frms_chunk*16,), np.float32)
        flag_gpu =   gpuarray.empty((n_frms_chunk*16,), np.uint16)
        niter_gpu =  gpuarray.empty((n_frms_chunk*16,), np.uint16)

        #
        # Open binary files to save the results into
        #
        basefn, t_stamp = cls.form_fout_name(cls.fname_m5b)
        basefn = basefn + "_" + t_stamp + ".bin"

        f_quantl = open("nt_hist_cuda_" + basefn, "wb")
        f_chi2 =   open("nt_chi2_cuda_" + basefn, "wb")
        f_thresh = open("nt_thresh_cuda_" + basefn, "wb")
        f_flag =   open("nt_flag_cuda_"  + basefn, "wb")
        f_niter =  open("nt_niter_cuda_"  + basefn, "wb")

        
        #
        # Main loop  CUDA  ======================================
        #

        for i_chunk in range(cls.n_m5b_chunks):

            tic = time.time()

            incompleteChunk = (i_chunk == cls.n_m5b_chunks-1) and \
                              (cls.n_frms_last_chunk != 0)
            
            #
            # Count chunk size and offset to read them one by one
            # from the m5b file
            #
            # Assume the current chunk is whole
            #
            n_words_chunk = cls.n_words_whole_chunk  
            n_bytes_chunk = cls.n_bytes_whole_chunk  
            n_words_last_chunk = cls.n_words_last_chunk
        
            n_frms = np.uint32(cls.n_frms_chunk)

            # However, the last chunk can be incomplete
            if incompleteChunk:
                n_words_chunk = n_words_last_chunk
                n_frms = np.uint32(cls.n_frms_last_chunk)

            n_words_chunk_offs = i_chunk * cls.n_words_whole_chunk
            n_bytes_chunk_offs = i_chunk * cls.n_bytes_whole_chunk
            
            #
            # Read a file chunk into the dat array
            #
            cls.dat = np.fromfile(cls.fname_m5b, dtype=np.uint32,
                                  count=n_words_chunk,
                                  offset=n_bytes_chunk_offs)
           
            toc = time.time()
            print("M5B file chunk has been read. Time: %7.3f s.\n" % (toc-tic))
            print("Chunk #%d, chunk size, words: %d, chunk offset, words: %d" %
                  (i_chunk, n_words_chunk, n_words_chunk_offs))

            #
            # Find how many CUDA blocks and CUDA threads per block needed
            #
            quot, rem = divmod(n_frms, cls.n_threads_max)
            if quot == 0:
                n_blocks = 1
                n_threads = rem
            elif rem == 0:
                n_blocks = quot
                n_threads = cls.n_threads_max
            else:   # Both quot and rem != 0:
                    # the last thread block will be < cls.n_threads_max 
                n_blocks = quot + 1
                n_threads = cls.n_threads_max

            n_blocks =  int(n_blocks)
            n_threads = int(n_threads)

            print('CUDA GPU Process Parameters:')
            print("Processing %d frames using %d CUDA blocks, "
                  "%d threads in each." % (n_frms, n_blocks, n_threads))
            if rem != 0:
                  print("The last, incomplete block has %d threads." % rem)

            tic = time.time()           

            #
            # Transfer host (CPU) memory to device (GPU) memory 
            #
            dat_gpu =     gpuarray.to_gpu(cls.dat)
            ch_mask_gpu = gpuarray.to_gpu(cls.ch_mask)

            #
            # For the last file chunk, if it is incomplete,
            # create new, smaller empty gpu arrays for the results
            #
            if incompleteChunk:
                n_frms = np.uint32(cls.n_frms_last_chunk)
                quantl_gpu = gpuarray.empty((n_frms*16*4,), np.float32)
                chi2_gpu =   gpuarray.empty((n_frms*16,), np.float32)
                thresh_gpu = gpuarray.empty((n_frms*16,), np.float32)
                flag_gpu =   gpuarray.empty((n_frms*16,), np.uint16)
                niter_gpu =  gpuarray.empty((n_frms*16,), np.uint16)

            #
            # Call the kernel on the CUDA GPU card
            #

            cls.m5b_gauss_test_cuda(dat_gpu, ch_mask_gpu,
                            quantl_gpu, chi2_gpu, thresh_gpu, flag_gpu,
                            niter_gpu, n_frms,
                            block = (n_threads, 1, 1), grid = (n_blocks, 1))

            #
            # Move the results from GPU memory to CPU RAM
            #
            quantl = quantl_gpu.get()
            chi2 =   chi2_gpu.get()
            thresh = thresh_gpu.get()
            flag =   flag_gpu.get()
            niter =  niter_gpu.get()

            del dat_gpu  # It occupies ~96% of the total array memory
            
            # del quantl_gpu
            # del chi2_gpu
            # del thresh_gpu
            # del flag_gpu
            # del niter_gpu
            # del ch_mask_gpu
        
            toc = time.time()
            print("\nGPU time: %.3f s." % (toc-tic))
            

            #
            # Save the results in binary files.
            # If several m5b file chunks are processed, the current chunk
            # data are appended to the file.
            #
            tic = time.time()

            quantl.tofile(f_quantl)
            chi2.tofile(f_chi2)
            thresh.tofile(f_thresh)
            flag.tofile(f_flag)
            niter.tofile(f_niter)
            
            toc = time.time()
            print('Saving a chunk of results in files: %.3f s.' % (toc-tic))

        f_quantl.close()
        f_chi2.close()
        f_thresh.close()
        f_flag.close()
        f_niter.close()
        
        #
        # Release GPU memory allocated to the large arrays -- needed, really??
        #
        del quantl_gpu
        del chi2_gpu
        del thresh_gpu
        del flag_gpu
        del niter_gpu
        del ch_mask_gpu
        
        tocg = time.time()
        
        tsec_total = tocg - ticg
        if tsec_total < 60.:
            print("\nTotal time: %.3f s." % (tocg-ticg))
        else:
            tminute = tsec_total // 60.0
            tsecond = tsec_total % 60.0
            print("\nTotal time: %d m, %.3f s (%.3f s)." % \
                  (tminute, tsecond, tsec_total))

        
        

    @classmethod
    def do_m5b_opencl(cls):

        ticg = time.time()

        cl = cls.cl    # OpenCL
        ctx = cls.ctx  # The context created at the class initialization
        
        n_threads_max = cls.n_threads_max
        n_frms = cls.n_frms_chunk
        
        mf = cl.mem_flags
        queue = cl.CommandQueue(ctx)

        #
        # Create output buffers in the CPU memory.
        #
        quantl = np.zeros((n_frms*16*4), dtype=np.float32) # Quantiles
        chi2 =   np.zeros((n_frms*16), dtype=np.float32)     # Chi^2
        thresh = np.zeros((n_frms*16), dtype=np.float32)   # Thresholds
        flag =   np.zeros((n_frms*16), dtype=np.uint16)    # Flags
        niter =  np.zeros((n_frms*16), dtype=np.uint16) # Iterations fminbndf()

        #
        # Create input and output buffers in the GPU memory.
        # The mf.COPY_HOST_PTR flag forces copying from the host buffer
        # to the device buffer referred as buf_) in the GPU memory.
        #
        buf_ch_mask = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=cls.ch_mask)
        buf_quantl = cl.Buffer(ctx, mf.WRITE_ONLY, quantl.nbytes)
        buf_chi2 = cl.Buffer(ctx, mf.WRITE_ONLY, chi2.nbytes)
        buf_thresh = cl.Buffer(ctx, mf.WRITE_ONLY, thresh.nbytes)
        buf_flag = cl.Buffer(ctx,   mf.WRITE_ONLY, flag.nbytes)
        buf_niter = cl.Buffer(ctx,  mf.WRITE_ONLY, niter.nbytes)


        #
        # Open binary files to save the results into
        #
        basefn, t_stamp = cls.form_fout_name(cls.fname_m5b)
        basefn = basefn + "_" + t_stamp + ".bin"

        f_quantl = open("nt_hist_opencl_" + basefn, "wb")
        f_chi2 =   open("nt_chi2_opencl_" + basefn, "wb")
        f_thresh = open("nt_thresh_opencl_" + basefn, "wb")
        f_flag =   open("nt_flag_opencl_"  + basefn, "wb")
        f_niter =  open("nt_niter_opencl_"  + basefn, "wb")

        
        #
        # Main loop OpenCL =========================================
        #

        for i_chunk in range(cls.n_m5b_chunks):

            tic = time.time()

            incompleteChunk = (i_chunk == cls.n_m5b_chunks-1) and \
                              (cls.n_frms_last_chunk != 0)
            
            #
            # Count chunk size and offset to read them one by one
            # from the m5b file
            #
            # Assume the current chunk is whole
            #
            n_words_chunk = cls.n_words_whole_chunk  
            n_bytes_chunk = cls.n_bytes_whole_chunk  
            n_words_last_chunk = cls.n_words_last_chunk
        
            n_frms = np.uint32(cls.n_frms_chunk)

            # However, the last chunk can be incomplete
            if incompleteChunk:
                n_words_chunk = n_words_last_chunk
                n_frms = np.uint32(cls.n_frms_last_chunk)

            n_words_chunk_offs = i_chunk * cls.n_words_whole_chunk
            n_bytes_chunk_offs = i_chunk * cls.n_bytes_whole_chunk
            
            #
            # Read a file chunk into the dat array
            #
            cls.dat = np.fromfile(cls.fname_m5b, dtype=np.uint32,
                                  count=n_words_chunk,
                                  offset=n_bytes_chunk_offs)

            toc = time.time()
            
            print("M5B file chunk has been read. Time: %7.3f s.\n" % (toc-tic))
            print("Chunk #%d, chunk size, words: %d, chunk offset, words: %d" %
                  (i_chunk, n_words_chunk, n_words_chunk_offs))

            #
            # Find how many Global Work Items, global_size (i.e. total number
            # of threads), are needed to process n_frms frames in the current
            # m5b file chunk provided that global_size must be a multiple
            # of the Work Group Size, local_size (i.e. number of threads per
            # block in CUDA terms). 
            #
            # In an OpenCL kernel: global_size = get_global_size(0)
            #
            # Find global_size >= n_frms as the least multiple of local_size
            # (i.e. work group size, which is n_threads_max per block here)
            #
            local_size = n_threads_max
            global_size = int(local_size*np.ceil(n_frms/local_size))
            nwg = global_size//local_size  # Number of work groups
            rem = n_frms % local_size

            print('OpenCL GPU Process Parameters:')
            print("Processing %d frames using %d OpenCL work groups, " \
                  "%d work items in each." % (n_frms, nwg, local_size))
            if rem != 0:
                print("The last, incomplete work group has %d work items." % \
                      rem)

            tic = time.time()           

            #
            # Transfer data chunk from host (CPU) memory to device (GPU) memory 
            #
            buf_dat = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=cls.dat)

            #
            # For the last m5b file chunk, if it is incomplete,
            # create new, smaller empty buffers for the results
            #
            if incompleteChunk:
                n_frms = np.uint32(cls.n_frms_last_chunk)
                
                quantl = np.zeros((n_frms*16*4), dtype=np.float32) # Quantiles
                chi2 = np.zeros((n_frms*16), dtype=np.float32)     # Chi^2
                thresh = np.zeros((n_frms*16), dtype=np.float32)   # Thresholds
                flag =   np.zeros((n_frms*16), dtype=np.uint16)    # Flags
                niter =  np.zeros((n_frms*16), dtype=np.uint16)    # Iterations
                
                buf_quantl = cl.Buffer(ctx, mf.WRITE_ONLY, quantl.nbytes)
                buf_chi2 = cl.Buffer(ctx, mf.WRITE_ONLY, chi2.nbytes)
                buf_thresh = cl.Buffer(ctx, mf.WRITE_ONLY, thresh.nbytes)
                buf_flag = cl.Buffer(ctx,   mf.WRITE_ONLY, flag.nbytes)
                buf_niter = cl.Buffer(ctx,  mf.WRITE_ONLY, niter.nbytes)
                
            #
            # Call the kernel on the OpenCL GPU card
            #

            cls.m5b_gauss_test_ocl(queue, (global_size,), (local_size,),
                             buf_dat, buf_ch_mask,  buf_quantl, buf_chi2, 
                             buf_thresh,  buf_flag, buf_niter,  n_frms).wait()

            #
            # Move the results from GPU memory to CPU RAM
            #
            cl.enqueue_copy(queue, quantl, buf_quantl)
            cl.enqueue_copy(queue, chi2, buf_chi2)
            cl.enqueue_copy(queue, thresh, buf_thresh)
            cl.enqueue_copy(queue, flag, buf_flag)
            cl.enqueue_copy(queue, niter, buf_niter)

            buf_dat.release()  # It occupies ~96% of the total array memory

            toc = time.time()
            print("\nGPU time: %.3f s." % (toc-tic))
            
            #
            # Save the results in binary files.
            # If several m5b file chunks are processed, this
            # appends the files with the results for current chunk
            #
            tic = time.time()

            quantl.tofile(f_quantl)
            chi2.tofile(f_chi2)
            thresh.tofile(f_thresh)
            flag.tofile(f_flag)
            niter.tofile(f_niter)
            
            toc = time.time()
            print('Saving a chunk of results in files: %.3f s.' % (toc-tic))


        queue.flush()
        queue.finish()

        buf_ch_mask.release()
        # buf_dat.release()
        buf_quantl.release()
        buf_chi2.release()
        buf_thresh.release()
        buf_flag.release()
        buf_niter.release()


        tocg = time.time()

        tsec_total = tocg - ticg
        if tsec_total < 60.:
            print("\nTotal time: %.3f s." % (tocg-ticg))
        else:
            tminute = tsec_total // 60.0
            tsecond = tsec_total % 60.0
            print("\nTotal time: %d m, %.3f s (%.3f s)." % \
                  (tminute, tsecond, tsec_total))






            

