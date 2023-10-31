NormTest

This software is intended for fast testing for correctness of the Mark 5B output
files (m5b files) before the correlator. The 2-bit data streams in the m5b files
can be considered correct if they are sampled from the Gaussian (normal)
distribution. We call such tests "testing for normality".

For complete description and documentation please refer to doc/benkevitch_normality_test_for_4_level_m5b_files.pdf. The doc directory also contains the LaTeX file benkevitch_normality_test_for_4_level_m5b_files.tex and the eps images for the figures therein.


1. gpu_m5b.py:

Normality (Gaussianity) test for M5B files on either an Nvidia GPU using PyCUDA
package or a GPU using PyOpenCL package. Single precision floats.

This module provides "transparent" access to the GPU independent of the
software framework used, CUDA or OpenCL.

The module contains class normtest. It is not intended to create multiple
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
in binary files.  The file have the following names:

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

   
Example:
   
from gpu_m5b import Normtest as nt
nt.do_m5b("rd1910_wz_268-1811.m5b")

Empirically, it has been found that the best performance is achieved 
with 8 threads per block (in CUDA terminology), and, which is the same, 
8 work items per work group (in OpenCL terms). However, this number can 
be changed using the nthreads parameter. For example:

nt.do_m5b("rd1910_wz_268-1811.m5b", nthreads=64)


---------------------------

This directory also contains two older versions ofthe Python scripts for
normality testing using the GPUs. One of them, normtest_m5b_cuda.py, uses
PyCuda and can be only run on Nvidia GPUs with CUDA and PyCuda installed.
The other one uses PyOpenCL and can be run on both AMD and Nvidia GPUs as
long as OpenCL and PyOpenCL are installed. Note that CUDA works ~1.5 times
faster than OpenCL.

Both scripts cannot process M5B files in chunks, and therefore they are unable
to process large M5B files.

Currently, the scripts use positional command line parameters.

2. normtest_m5b_cuda.py:

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




3. normtest_m5b_ocl.py:

Normality (Gaussianity) test for M5B files on a GPU using PyOpenCL package.
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
")
$ python normtest_m5b_ocl.py rd1910_wz_268-1811.m5b
$ python normtest_m5b_ocl.py rd1910_wz_268-1811.m5b 8
$ python normtest_m5b_ocl.py rd1910_wz_268-1811.m5b 8 -s

Note that saving the result files may take long time.


----------------------------

Ancillary scripts:

4. plot_m5b_thrange.py

Plots squared error (the residual) between M5B Data and four quantiles of the
standard normal distribution over the range [0.4 .. 1.5] of input sample
thresholds. The plot obviously has its minimum at the threshold +-0.817/STD
marked with the red dot.


5. test_m5b.py

Calculates the observed Chi^2 and compares it with the Chi^2 critical value at
the significance level 0.05 for the degrees of freedom 3 (we have 4 quantiles,
so df = 4 - 1 = 3).

Plots histograms of the observed data and the theoretical normal distribution
to compare.

Running:
%run test_m5b.py <m5b_filename>


6. plot_m5b_hist.py

Plots two histograms of the results from gpu_m5b.py for the whole 
M5B (or M5A) file: 
6.1. Distribution of chi^2 and a red marker showing position of the critical
     chi^2 value (7.81), as well as the percent of chi^2 exceeding it.
6.2. Distribution of the optimal quantization thresholds and a red marker
     showing position of the critical threshold value (0.6745 rms), as well as
     the percent of the thresholds that failed to reach it.


7. inspect_nt.py

This script creates 4x4 plots of 16 histograms for each of the 16 channels.
The plots are for one or several (averaged) frames. The histograms are compared
with the normal distribution curves showing approximately from what size
quantiles the observation data are drawn. For each plot the chi^2 is printed.

The data are read from the *.bin files created with the gpu_m5b_chi2.py.

Running:

%run inspect_nt.py <m5b_filename> <timestamp> <start_frame_#> <#_of_frames> 

Some interesting frames:
7.1. These plots show that the Pearson's chi^2 cannot be used. The histograms 
are very far from the normal distributions, but most of the chi^2 values
are very small and signal "normality"
%run inspect_nt.py rd1903_ft_100-0950.m5b 025 97737 1

7.2. These plots are definitely from the uniform distributions, and yet, most
of the chi^2 values are very small and signal "normality". The pearson's
criterion also does not work.
%run inspect_nt.py rd1903_ft_100-0950.m5b 025 4 1
%run inspect_nt.py rd1903_ft_100-0950.m5b 025 170578 1
%run inspect_nt.py rd1903_ft_100-0950.m5b 025 389930 1
%run inspect_nt.py rd1903_ft_100-0950.m5b 025 6832715 1

7.3. These plots show close to normal histograms with good chi^2 values, i.e.
< 7.81.
%run inspect_nt.py rd1910_wz_268-1811.m5b 970 200 1
%run inspect_nt.py rd1910_ny_269-1404.m5a 395 7139 1


8. plot_25pc_npdf_expectations.py

Plots the normal curve N(0,1) divided vertically into 4 equal areas under the 
curve (25%-quantiles). The division lines are at -0.6745, 0, and +0.6745. 
For each of the areas, the math expectations are computed, they are at 
    -1.27, -0.32, 0.32, and 1.27 (in STDs). 
Thse are the most probable analog signal values before the quantization with
the *ideal* thresholds -0.6745, 0, and +0.6745, which provide the uniform 
arrangement of the discrete signal in the 4 bins.  


9. skew_kurt.py

Calculates skewness and kurtosis of a frame in M5B or M5A file.
The data positions (in STDs) are assumed at the 25%-quantile math expectations,
  mu0 = -1.27, mu1 = -0.32, mu2 = 0.32, and mu3 = 1.27.



