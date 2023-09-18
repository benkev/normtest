help_txt = '''
plot_chi2.py
'''

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as pl

pl.ion()  # Interactive mode; pl.ioff() - revert to non-interactive.
pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

#
# alpha: level of significance
# df: number of degrees of freedom
#
alpha = 0.05    # level of significance
df = 3          # number of degrees of freedom

chi2cr = chi2.ppf(1-alpha, df)

print('Chi^2 critical value at significance %.2f and df = %d: %.2f' % \
      (alpha, 3, chi2cr))


x = np.linspace(chi2.ppf(0.001, df), chi2.ppf(0.999, df), 100)
x095 = np.linspace(chi2cr, chi2.ppf(0.999, df), 20)


#pl.figure()
pl.plot(x, chi2.pdf(x, df), 'k', label='chi2 pdf'); pl.grid(1)
pl.fill_between(x095, chi2.pdf(x095, df), color='r')
pl.plot([chi2cr,chi2cr], [0,0.095], 'k--', lw=0.9)

pl.plot([-0.2, x[-1]], [0,0], 'k', lw=0.9)    # X axis
pl.plot([0,0], [-0.005, 0.25], 'k', lw=0.9)  # Y axis

pl.text(chi2cr-0.1, 0.105, r"$\chi^2_{cr}=$ %4.2f" % chi2cr, fontsize=14)
pl.text(2.6, 0.03, r"$p=%.2f$" % (1-alpha), fontsize=12)
pl.text(10.2, 0.03, r"$p=%.2f$" % alpha, fontsize=12)

pl.title(r"Probability Density Function of $\chi^2_{k=%d}$" % df, fontsize=14)

pl.xlabel(r"$\chi^2_{k=%d}$" % df, fontsize=12)
pl.ylabel(r"$f_{k=%d}(x)$" % df, fontsize=12)

pl.show()

