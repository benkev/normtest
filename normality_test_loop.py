#
# normality_test_loop.py
#

import numpy as np
import matplotlib.pyplot as pl
import scipy.stats

from numpy.random import seed
from numpy.random import normal

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

N = int(1e4)
h = 0.25                 # Interval length
alpha = 0.95
# seed(12)

for itr in range(100):
    dat_mean = np.round(np.random.normal(1, 2), 1)
    dat_sig = np.round(np.random.uniform(0.5, 3), 1)

    dat = normal(loc=dat_mean, scale=dat_sig, size=N)

    lx = np.floor(dat.min()) # Left interval limit
    rx = np.ceil(dat.max())  # Right interval limit
    nedges = int((rx - lx)/h) + 1
    edges = np.linspace(lx, rx, nedges)    # Bin edges
    x = edges[:-1] + 0.5*h                 # Bin middle points
    frq, binedg = np.histogram(dat, bins=edges)
    nfrq = len(frq)

    xmean = np.sum(x*frq)/N
    xsig = np.sqrt(np.sum((x**2)*frq)/N - xmean**2)
    z = (x - xmean)/xsig  # Standardized x
    normal_PDF = np.exp(-z**2/2)/np.sqrt(2*np.pi) # Standard normal PDF
    tfrq = (h*N/xsig)*normal_PDF  # Theoretical frequency
    chi2 = np.sum((frq - tfrq)**2/tfrq)

    # raise SystemExit

    #
    # Critical value of chi2(k) with k degrees of freedom, for significance 0.95
    # dat is normal (at significance alpha) if chi2 < chi2cr
    #
    k = nfrq - 2 - 1
    chi2cr = scipy.stats.chi2.ppf(alpha, k)



    cmps = ' < ' if chi2 < chi2cr else ' > '
    norm_or_not = ' Normal ' if chi2 < chi2cr else ' NOT Normal '

    pl.figure();
    pl.plot(x, frq, '.', color='blue', label='Observed'); pl.grid(1)
    pl.hist(dat, bins=edges, alpha=0.2, color='green', histtype='stepfilled')
    pl.plot(x, tfrq, color='red', label='Theoretical')
    pl.legend(fontsize=15, loc='upper right')
    pl.title('Pearson Normality test: $N_x$ = %d, $\emph{bin}$ = %g' % \
             (N, h), fontsize=15)
    pl.xlabel('$x$', fontsize=18)
    pl.ylabel('Frequency', fontsize=15)

    pl.figtext(0.14, 0.82, r'$\chi^2_{obs} =$ %6.2f' % chi2, fontsize=15)
    pl.figtext(0.14, 0.76, r'$\chi^2_{crit} =$ %6.2f' % chi2cr, fontsize=15)
    pl.figtext(0.14, 0.70, r'$\chi^2_{obs} %s \chi^2_{crit}:$' % \
               cmps, fontsize=15)
    pl.figtext(0.14, 0.63, r'%s' % norm_or_not, color='red', fontsize=20)
    pl.figtext(0.14, 0.58, r'significance %4.2f' % alpha, fontsize=15)
    pl.figtext(0.14, 0.52, r'deg. of freedom %d' % k, fontsize=15)

    pl.figtext(0.66, 0.63, r'$mean = %5.2f$' % dat_mean, fontsize=15)
    pl.figtext(0.66, 0.58, r'$\sigma=%5.2f$' % dat_sig, fontsize=15)
    pl.figtext(0.66, 0.52, r'$x \in [%d  ..  %d]$' % (lx, rx), fontsize=15)

    pl.savefig('fig/Pearson_test_N%d_bin%g_mean%g_std%g.svg' % \
               (N, h, dat_mean, dat_sig), format='svg')

    # pl.show()
    pl.close()

    print()
    print('N = ', N, ', nfrq = ', nfrq)
    print('x in [%d .. %d], bin size = %6.4f' % (lx, rx, h))
    print('chi2_observed = %6.2f' % chi2, ', chi2_critical = %6.2f' % chi2cr, \
          ' at k = %d degrees of freedom' % k)
    print('chi2_observed' + cmps + 'chi2_critical: ' + \
          norm_or_not + ' at %4.2f significance level' % alpha)



