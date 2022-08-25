import numpy as np
import importlib
import os, sys
# import time

class normtest:

    name = 'normtest'
    fmega = pow(1024.,2)
    fgiga = pow(1024.,3)
    

    #
    # M5B file parameters
    #
    n_frmwords = 2504;   # 32-bit words in one frame including the 4-word header
    n_frmbytes = 2504*4  # Bytes in a frame
    n_frmdatwords = 2500; # 32-bit words of data in one frame

    quota_dat = 0.95  # Quota of dat array in overall GPU data (approx)

    #
    # Determine if PyCUDA or/and PyOpenCL are installed
    #
    pycuda_installed = importlib.util.find_spec("pycuda") is not None
    pyopencl_installed = importlib.util.find_spec("pyopencl") is not None

    #
    # In case both are installed, prefer PyCUDA since it is ~1.5 times faster
    #
    # use_pycyda = False
    # use_pyopencl = False
    
    gpu_framework = ""
    if pycuda_installed:
        gpu_framework = "cuda"     # Use PyCUDA even if PyOpenCL installed
    elif pyopencl_installed:
        gpu_framework = "opencl"   # Use PyOpenCL if PyCUDA not installed
    else:
        sys.exit("Neither PyCUDA nor PyOpenCL are installed. Exiting.")

    import psutil
    mem = psutil.virtual_memory() 
    mem_cpu_total = mem.total
    mem_cpu_available = mem.available
    mem_cpu_used = mem.used
    mem_cpu = mem_cpu_total  # Use total of CPU memory
    sz_cpu = mem_cpu_total   # Use total of CPU memory



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
        platname = plat.name
        platname = platname.split()[0] # Get the first word

        if platname == 'NVIDIA':
            ker_opts = ['-I . -D __nvidia']
        elif platname == 'AMD':
            ker_opts = ['-I . -D __amd']
        else:
            sys.exit("Platform ""%s"" is not supported. Exiting." % platname)

        devs = plat.get_devices(cl.device_type.ALL)
        ocl_dev = devs[0]
        #
        # No idea how to get _available_ global memory in OpenCL !!!!!!!!!!!!!!
        #
        mem_gpu_total = ocl_dev.global_mem_size 
        sz_gpu = mem_gpu_total   # Use all the available GPU global memory
        # sz_gpu = mem_gpu_free  # Only use the available GPU global memory ???
    
        mf = cl.mem_flags
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
    def do_m5b(cls, fname_m5b):
        
        n_m5bbytes = os.path.getsize(fname_m5b)
        n_m5bwords = n_m5bbytes // 4
        cls.sz_m5b = n_m5bbytes
        n_whole_frms = n_m5bbytes // cls.n_frmbytes
        n_whole_words = cls.n_frmwords*n_whole_frms
        n_last_frmbytes = n_m5bbytes % cls.n_frmbytes
        n_last_frmwords = n_last_frmbytes // 4
        '''
        Select processing mode dependent on relation sz_m5b ~ sz_cpu ~ sz_gpu
        
        If f == sz_m5b  (M5B file size),
           c == sz_cpu  (CPU memory size),
           g == sz_gpu  (GPU memory size),
        then there are 6 = 3! options:
          f < c < g
          f < g < c
          c < f < g
          g < f < c
          g < c < f
          c < g < f        

        '''
        f = sz_m5b = cls.sz_m5b
        c = sz_cpu = cls.sz_cpu
        g = sz_gpu = cls.sz_gpu

        fitsBoth = (f < c) and (f < g) # M5B file fits both CPU and GPU
        fitsNone = (f > c) and (f > g) # M5B file fits neither CPU nor GPU
        fitsCPUnotGPU = g < f < с      # M5B file does not fit CPU but fits GPU
        fitsGPUnotCPU = c < f < g      # M5B file does not fit GPU but fits CPU

        cls.fitsBoth = fitsBoth
        cls.fitsNone = fitsNone
        cls.fitsGPUnotCPU = fitsGPUnotCPU   # WE DO NOT USE IT
        cls.fitsCPUnotGPU = fitsCPUnotGPU

        #
        # Determine the M5B file partitioning: how many frames it contains,
        # how many whole frames. How many whole frames fit a chunk size to read
        # and process in one read. What is the size of the last, incomplete
        # chunk (if there is one). What are the chunk and read offsetts sizes
        # in uint32 words.
        #
        # Assume the file chunk to read into dat array is ~95% of available
        # GPU memory (quota_dat = .95)
        #
        # Maximum dat array size (bytes) to fit GPU along with other arrays 
        sz_dat_max = int(cls.quota_dat*sz_gpu)
        # Number of the whole frames in a chunk to read at once
        # to fit GPU memory
        n_frms_chunk = sz_dat_max // cls.n_frmbytes
        # The size (bytes) of dat array composed of whole frames to fit GPU
        # along with other arrays
        sz_dat = n_frms_chunk*cls.n_frmbytes
        # Size of the file chunk of whole frames to read at once in 32bit words
        n_words_chunk = n_frms_chunk*cls.n_frmwords
        # Number of whole chunks (each having n_frms_chunk of whole frames) in
        # the entire m5b file
        n_m5b_whole_chunks = n_m5bwords // n_words_chunk
        # Number of uint32 words in the last, incomplete chunk of the m5b file
        n_words_last_chunk = n_m5bwords % n_words_chunk
        # Determine how many whole frames the last chunk has
        n_frms_last_chunk = n_words_last_chunk // cls.n_frmwords
        # Number of words in the last (incomplete?) chunk to make a whole
        # number of frames
        n_words_last_chunk_whole_frms = n_frms_last_chunk*cls.n_frmwords

        #
        # Create the lists of all chunk sizes and all the corresponding
        # offsetts to read them one by one from the m5b file
        #
        chunk_size_words = [n_words_chunk for i in range(n_m5b_whole_chunks)]
        n_m5b_chunks = n_m5b_whole_chunks  # Assume all chunks are whole 
        if n_words_last_chunk_whole_frms != 0:
            chunk_size_words += [n_words_last_chunk_whole_frms]
            n_m5b_chunks = n_m5b_whole_chunks + 1 # Include the incomplete chunk
        chunk_offs_words = [i*n_words_chunk for i in range(n_m5b_chunks)]
            

        #
        # Main loop =====================================================
        #


        print("n_m5bbytes = ", n_m5bbytes, ", n_m5bwords = ", n_m5bwords)
        print("n_words_chunk = ", n_words_chunk)
        print("n_frms_chunk = ", n_frms_chunk)
        print("chunk_size_words = ", chunk_size_words)
        print("chunk_offs_words = ", chunk_offs_words)

        
        for i_chunk in range(n_m5b_chunks):
            
            cls.dat = np.fromfile(fname_m5b, dtype=np.uint32,
                                  count=chunk_size_words[i_chunk],
                                  offset=chunk_offs_words[i_chunk])
            print("chunk_size_words[i_chunk] = ", chunk_size_words[i_chunk])
            print("chunk_offs_words[i_chunk] = ", chunk_offs_words[i_chunk])
        
        

        # if fitsBoth or fitsCPUnotGPU:
        #     cls.dat = np.fromfile(fname_m5b, dtype=np.uint32,
        #                           count=n_chunkwords, offset=offs)
            
        # if cls.gpu_framework == "cuda":
        #     cls.m5b_gauss_test_cuda(dat_gpu, ch_mask_gpu,
        #                     quantl_gpu, residl_gpu, 
        #                     thresh_gpu,  flag_gpu, niter_gpu,  n_frms,
        #                     block = (Nthreads, 1, 1), grid = (Nblocks, 1))

        # if cls.gpu_framework == "opencl":
        #     cls.m5b_gauss_test_ocl(queue, (wiglobal,), (wgsize,),
        #                    buf_dat, buf_ch_mask,  buf_quantl, buf_residl, 
        #                    buf_thresh,  buf_flag, buf_niter,  n_frms).wait()
        
            
    
    # def __init__(self, a, b):
    #     self.a = a
    #     self.b = b 
    
    # @classmethod
    # def info(cls):
    #     return cls.name


        # if (f < c) and (f < g):   # M5B file fits both CPU and GPU
        #     cls.dat = np.fromfile(fname_m5b, dtype=np.uint32,
        #                       count=cls.n_frmwords*total_frms)
        
        # if c < f < g:   # M5B file does not fit GPU but fits CPU
        #     pass

        # if g < f < с:   # M5B file does not fit CPU but fits GPU
        #     pass

        # if (f > c) and (f > g):   # M5B file fits neither CPU nor GPU
        #     pass
            
    
