#
# chi2.py
#
# 
#

import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
from scipy.stats import chi2


fnorm = lambda x: (1/np.sqrt(2*np.pi))*(np.exp(-0.5*x**2))  # Normal PDF (0,1)
Fnorm = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))               # Normal CDF (0,1)


#
# Degrees of freedom, df
#
# For each frame we have 4 random values. They are bound by 2 restrictions:
# (1) Normalizing factor N = 2500, and
# (2) Quantization threshold found from the observed numbers.
# Therefore, the number of degrees of freedom for chi^2 distribution
# is 2 = 4 - 2.
#
df = 2

x = np.linspace(0, 15, 100)

# pl.figure();
# pl.plot(x, chi2.pdf(x, 2), label="PDF $\chi^2_2(x)$"); pl.grid(1)
# pl.plot(x, chi2.cdf(x, 2), label="CDF $\chi^2_2(x)$");
# pl.legend()


pl.figure()
pl.plot(x, 1 - chi2.cdf(x, df=df)); pl.grid(1)
pl.title(r"Survival Function $1 - \mathrm{CDF}[\chi^2(%d,x)]$" % df)
pl.xlabel(r"$x$")
pl.ylabel(r"$p$")

#
# chi2.ppf and chi2.cdf are inverse of each-other:
#
# from scipy.stats.distributions import chi2
# chi2.ppf(0.95, df=5)     # 11.07
# chi2.cdf(11.07, df=5)    # 0.95
#
# chi2.isf(q, *args, **kwds)
# Inverse survival function (inverse of `sf`) at q of the given RV (Random
# Variable)
#
# from scipy import stats
# upperv=stats.chi2.isf(1-alpha/2,nu)
#
# Quantile corresponding to the upper tail probability q.
#


y = np.logspace(-3, 0, 100)
# pl.figure(); pl.plot(y, chi2.ppf(y, 2)); pl.grid(1)
# pl.title(r"$\mathrm{PPF}[\chi^2_\nu(2,x)]$")
# pl.xlabel(r"$x$")

pval = np.array([0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10,
                 0.05, 0.01, 0.001])
pl.figure();
pl.plot(y, chi2.isf(y, df=2)); pl.grid(1)
pl.plot(pval, chi2.isf(pval, df=2), 'r.')
pl.title(r"Inverse Survival Function $\mathrm{ISF}[\chi^2(%d,x)]$" % df)
pl.xlabel(r"$p$")
pl.ylabel(r"$x$")

print()
print("   p    chi2.isf(p, df=2)")
for p in pval:
    #print(1-p, chi2.ppf(p, df=2))
    print("%.4f  %f" % (p, chi2.isf(p, df=2)))
print()

# for p in pval:
#     print("%f  %f" % (p, chi2.ppf(1-p, df=2)))
#     #print("%f  %f" % (p, chi2.isf(p, df=4)))


# print("Critical value of $\chi^2_2(x)$ for $p_{value}= 0.05$: %f" % \
#       chi2.isf(0.05, df=2))
print("Critical value of chi^2(2,x) for p-value = 0.05: %f" % \
      chi2.isf(0.05, df=2))



pl.show()



