help_text = '''

plot_25pc_npdf_expectations.py

Plots the normal curve N(0,1) divided vertically into 4 equal areas under the 
curve (25% quantiles). The division lines are at -0.6745, 0, and +0.6745. 
For each of the areas, the math expectations are computed, they are at 
    -1.27, -0.32, 0.32, and 1.27 (in STDs). 
Thse are the most probable analog signal values before the quantization with
the *ideal* thresholds -0.6745, 0, and +0.6745, which provide the uniform 
arrangement of the discrete signal in the 4 bins.  

'''

import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
import scipy.stats
import scipy.integrate as integrate
import getopt, glob

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

ncdf =  scipy.stats.norm.cdf
npdf =  scipy.stats.norm.pdf

N = 2500       # Measurements in one m5b data frame (per channel)

# fnorm = lambda x: (1/np.sqrt(2*np.pi))*(np.exp(-0.5*x**2))  # Normal PDF (0,1)
# Fnorm = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))               # Normal CDF (0,1)

import scipy.stats as stats

def interval_mean(a, b, mu=0, sigma=1):
    # Define the weighted function (x * PDF)
    f = lambda x: x*npdf(x, mu, sigma)
    
    # Integrate the weighted function and the PDF over the interval
    integral_f = integrate.quad(f, a, b)[0]
    integral_pdf = integrate.quad(lambda x: npdf(x, mu, sigma), a, b)[0]
    
    return integral_f / integral_pdf


# -a2 = interval_mean(-1000, -0.6745)
# -a1 = interval_mean(-0.6745, 0)
a1 = interval_mean(0, 0.6745)
a2 = interval_mean(0.6745, 1000)

print("-Inf .. -0.6745: interval expectation = ", -a2)
print("-0.6745 .. 0:    interval expectation = ", -a1)
print("0 .. 0.6745:     interval expectation = ",  a1)
print("0.6745 .. +Inf:  interval expectation = ",  a2)

xrul = np.linspace(-3.3, 3.3, 201)

# th = np.array([-0.6745, 0, 0.6745])
th = 0.6745
aa = np.array([-a2, -a1, a1, a2]) # Interval means of the quantiles
npdfa = npdf(aa) 

pl.ion()

pl.figure()

pl.plot(xrul, npdf(xrul)); pl.grid(1)
# pl.plot([-a2, -a1, a1, a2], [npdf(-a2), npdf(-a1), npdf(a1), npdf(a2)], "ro")
pl.plot([-th, -th], [0, 1.05*npdf(-th)], "k--")
pl.plot([0, 0], [0, 1.05*npdf(0)], "k--", label="25\% areas")
pl.plot([th, th], [0, 1.05*npdf(th)], "k--")
pl.plot(aa, npdfa, "ro", markersize=4, label="Interval Means")
pl.plot(aa, [0,0,0,0], "ro", markersize=4)
pl.plot([-a2,-a2], [0,npdf(-a2)],  "r-.", lw=0.5)
pl.plot([-a1,-a1], [0,npdf(-a1)],  "r-.", lw=0.5)
pl.plot([a1,a1], [0,npdf(a1)],  "r-.", lw=0.5)
pl.plot([a2,a2], [0,npdf(a2)],  "r-.", lw=0.5)


# pl.plot(xrul, f_pdf, 'orange')
pl.fill_between(xrul, npdf(xrul), where=(-3.3<xrul)&(xrul<=-th),
                color='b', alpha=0.2)
pl.fill_between(xrul, npdf(xrul), where=(-th<xrul)&(xrul<=0),
                color='m', alpha=0.2);
pl.fill_between(xrul, npdf(xrul), where=(0<xrul)&(xrul<=th),
                color='g', alpha=0.2)
pl.fill_between(xrul, npdf(xrul), where=((th<xrul)&(xrul<10)),
                color='c', alpha=0.2)

pl.xticks([-a2, -a1, a1, a2], ['-1.27', '-0.32', '0.32', '1.27'], fontsize=12)

pl.legend(loc='upper right', fontsize=12)

pl.annotate("-0.6745", (-0.6745, 0.330), (-1.5, 0.34), fontsize=12)
pl.annotate("0.6745", ( 0.6745, 0.330), (0.6745, 0.34), fontsize=12)
# pl.annotate(r"$-\mu_2$", (-1.8, 0.18), (-1.8, 0.18), color='r')
# pl.annotate(r"$\mu_2$",  ( 1.45, 0.18), ( 1.45, 0.18), color='r')
# pl.annotate(r"$-\mu_1$", (-0.8,0.38), (-0.8,0.38), color='r')
pl.text(-1.8, 0.18, r"$-\mu_2$", color='r', fontsize=12)
pl.text(1.45, 0.18, r"$\mu_2$", color='r', fontsize=12)
pl.text(-0.8, 0.38, r"$-\mu_1$", color='r', fontsize=12)
pl.text( 0.45, 0.38, r"$\mu_1$", color='r', fontsize=12)

xl = pl.xlim()
x0 = xl[0] + 0.015*(xl[1] - xl[0])
pl.annotate(r"$\mu_{[a,b]} = \frac{\int_a^b x f(x) dx}{\int_a^b f(x) dx}$",
            (x0, 0.24), (x0, 0.24), fontsize=17)

pl.figtext(0.5, 0.92, r"25\%-Quantiles of Normal PDF and Their Expectations " \
           "$-\mu_2, -\mu_1, \mu_1, \mu_2$", fontsize=14, ha='center')

pl.show()
