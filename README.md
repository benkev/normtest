NormTest

This software is intended for fast testing for correctness of the Mark 5B output
files (m5b files) before the correlator. The 2-bit data streams in the m5b files
can be considered correct if they are samples from the Gaussian (normal)
distribution. We call such tests "testing for normality".

This directory contains two Python scripts for normality testing using the
GPUs. One of them, normtest_m5b_cuda.py, uses PyCuda and can be only run on
Nvidia GPUs with CUDA and PyCuda installed. The other one uses PyOpenCL and can
be run on both AMD and Nvidia GPUs as long as OpenCL and PyOpenCL are
installed. Note that CUDA works ~1.5 times faster than OpenCL.

Currently, the scripts use positional command line parameters.

1. normtest_m5b_cuda.py:

Normality (Gaussianity) test for M5B files on Nvidia GPU using PyCUDA package.
Single precision floats.

Requires:

ker_m5b_gauss_test.cu, CUDA kernel.
fminbndf.cu, 1D optimization code.

Usage:

$ python normtest_m5b_cuda.py <m5b-file-name> [<of threads per block>] [-s]

Or, from IPython:

$ ipython --pylab

%run normtest_m5b_cuda.py <m5b-file-name> [<# of threads per block>] [-s]

If <# of threads> is not specified, the optimal (appearingly) 8 is used.

If "-s" is present at the end of command line the results are saved in text
files in the subdirectory result/ with the names:
   thresholds_*.txt
   residuals_*.txt
   n_iterations_*.txt
   flags_*.txt
   quantiles_*.txt

Examples:

$ python normtest_m5b_cuda.py rd1910_wz_268-1811.m5b
$ python normtest_m5b_cuda.py rd1910_wz_268-1811.m5b 8
$ python normtest_m5b_cuda.py rd1910_wz_268-1811.m5b 8 -s

Note that saving the result files may take long time.




2. normtest_m5b_ocl.py:

Normality (Gaussianity) test for M5B files on GPU using PyOpenCL package.
Single precision floats.

Requires:
ker_m5b_gauss_test.cl, OpenCL kernel.
fminbndf.cl, 1D optimization code for Nvidia GPUs.
fminbndf_amd.cl, 1D optimization code for AMD GPUs.

Usage: 

$ python normtest_m5b_cuda.py <m5b-file-name> [<# of threads per block>] [-s]

Or, from IPython:

$ ipython --pylab

%run normtest_m5b_ocl_.py <m5b-file-name> [<of threads per block>] [-s]

If <# of threads> is not specified, the optimal (appearingly) 8 is used.

If "-s" is present at the end of command line the results are saved in text
files in the subdirectory result/ with the names:
   thresholds_*.txt
   residuals_*.txt
   n_iterations_*.txt
   flags_*.txt
   quantiles_*.txt

Examples:

$ python normtest_m5b_ocl.py rd1910_wz_268-1811.m5b
$ python normtest_m5b_ocl.py rd1910_wz_268-1811.m5b 8
$ python normtest_m5b_ocl.py rd1910_wz_268-1811.m5b 8 -s

Note that saving the result files may take long time.


3. gpu_m5b.py:
   
This module contains class normtest. It is not intended to create multiple
class instances (althoug it is surely possible). When imported, it 
probes the system to find what GPU frameworks are installed. It chooses
automatically between PyCUDA and OpenCL and initializes the relevant 
data structures.

The normtest class provides a "class method" do_m5b(m5b_filename),
which runs the normality test on the available GPU and the software
framework selected. 
   
Example:
   
from gpu_m5b import normtest as nt
thres, resid, n_iter, flag, quantl = nt.do_m5b("rd1910_wz_268-1811.m5b")








 
