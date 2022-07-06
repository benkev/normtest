import pyopencl as cl
import sys

argv = sys.argv
if len(argv) > 1:
    ker_name = argv[1]
else:
    ker_name = "ker_m5b_gauss_test_amd.cl"


ctx = cl.create_some_context()

#
# Read the kernel code from file into the string "ker"
#
with open (ker_name) as fh: ker = fh.read()

print("Compiling " + ker_name + ":") 

prg = cl.Program(ctx, ker).build(options=['-I .'])

