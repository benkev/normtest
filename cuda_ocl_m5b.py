def cuda_m5b(dat, ch_mask, ker_code):

    import pycuda as cu
    import pycuda.gpuarray as gpuarray
    import pycuda.compiler as compiler

    #
    # Initialize the device (instead of import pycuda.autoinit)
    #
    cu.driver.init()

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
    flag_gpu =   gpuarray.empty((nfrm*16,), np.uint16)
    niter_gpu =  gpuarray.empty((nfrm*16,), np.uint16)

    #
    # Compile the kernel code 
    #
    mod = compiler.SourceModule(ker_code,
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

    return residl, thresh, quantl, niter, flag





def opencl_m5b(dat, ch_mask, ker_code):

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
        print('Platform "%s" is not supported. Exiting.' % platname)
        raise SystemExit


    mf = cl.mem_flags
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    #
    # Create input () and output ()
    # buffers in the GPU memory. The mf.COPY_HOST_PTR flag forces copying from
    # the host buffer, , to the device buffer (referred as buf_)
    # in the GPU memory.
    #
    buf_ch_mask = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=ch_mask)
    buf_dat = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dat)
    buf_quantl = cl.Buffer(ctx, mf.WRITE_ONLY, quantl.nbytes)
    buf_residl = cl.Buffer(ctx, mf.WRITE_ONLY, residl.nbytes)
    buf_thresh = cl.Buffer(ctx, mf.WRITE_ONLY, thresh.nbytes)
    buf_flag = cl.Buffer(ctx,   mf.WRITE_ONLY, flag.nbytes)
    buf_niter = cl.Buffer(ctx,  mf.WRITE_ONLY, niter.nbytes)

    prg = cl.Program(ctx, ker_code).build(options=ker_opts)

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

    quantl = quantl.reshape(nfrm,16,4)
    residl = residl.reshape(nfrm,16)
    thresh = thresh.reshape(nfrm,16)
    flag = flag.reshape(nfrm,16)
    niter = niter.reshape(nfrm,16)

    return residl, thresh, quantl, niter, flag







