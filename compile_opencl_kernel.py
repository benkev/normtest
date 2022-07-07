#
# compile_opencl_kernel.py
#
# Run OpenCL compiler to elicit syntacs/semantic errors, if any.
#
# Examples:
#
# $ python compile_opencl_kernel.py ker_m5b_gauss_test_amd.cl
# $ python compile_opencl_kernel.py ker_m5b_gauss_test_nvidia.cl
#
# Or, from IPython:
#
# $ ipython --pylab
# $ In [1]: %run compile_opencl_kernel.py ker_m5b_gauss_test_amd.cl
# $ In [2]: %run compile_opencl_kernel.py ker_m5b_gauss_test_nvidia.cl
#

import pyopencl as cl
import sys

argv = sys.argv
if len(argv) > 1:
    ker_name = argv[1]
else:
    print('Error: an OpenCL kernel file must be specified. Quitting.')
    # ker_name = "ker_m5b_gauss_test_amd.cl"
    # ker_name = "ker_m5b_gauss_test_amd.cl"


ctx = cl.create_some_context()

#
# Read the kernel code from file into the string "ker"
#
with open (ker_name) as fh: ker = fh.read()

print("Compiling " + ker_name + ":") 

prg = cl.Program(ctx, ker).build(options=['-I .'])

