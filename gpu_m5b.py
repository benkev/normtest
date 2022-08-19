import importlib
import os, sys

class normtest:

    name = 'normtest'
    fmega = pow(1024.,2)
    fgiga = pow(1024.,3)

    

    #
    # M5B file parameters
    #
    frmwords = 2504;   # 32-bit words in one frame including the 4-word header
    frmbytes = 2504*4  # Bytes in a frame
    nfdat = 2500;      # 32-bit words of data in one frame


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
        # sz_gpu = mem_gpu_free  # Only use the available GPU global memory
        sz_gpu = mem_gpu_total   # Use all the available GPU global memory

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

        m5bbytes = os.path.getsize(fname_m5b)
        zs_m5b = m5bbytes
        
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

        if gpu_framework == "cuda":
            m5b_gauss_test(dat_gpu, ch_mask_gpu,  quantl_gpu, residl_gpu, 
                           thresh_gpu,  flag_gpu, niter_gpu,  nfrm,
                           block = (Nthreads, 1, 1), grid = (Nblocks, 1))

        if gpu_framework == "opencl":
            m5b_gauss_test(queue, (wiglobal,), (wgsize,),
                           buf_dat, buf_ch_mask,  buf_quantl, buf_residl, 
                           buf_thresh,  buf_flag, buf_niter,  nfrm).wait()
        
            
    
    # def __init__(self, a, b):
    #     self.a = a
    #     self.b = b 
    
    # @classmethod
    # def info(cls):
    #     return cls.name
