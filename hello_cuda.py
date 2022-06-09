#!python 
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void dot(int *result, int *a, int *b)
{
  const int i = threadIdx.x;
  result = result+ a[i] * b[i];
}
""")

#multiply_them = mod.get_function("dot")
dot = mod.get_function("dot")

a = numpy.random.randint(1,20,5)
b = numpy.random.randint(1,20,5) 
result = 0
dot(
        drv.Out(result), drv.In(a), drv.In(b),
        block=(5,1,1))

print(result)
