import pyopencl as cl

ctx = cl.create_some_context()

#
# Read the kernel code from file into the string "ker"
#
with open ("ker_m5b_gauss_test.cl") as fh: ker = fh.read()

prg = cl.Program(ctx, ker).build(options=['-I .'])


