#
# normality_test.py
#

import numpy as np
import matplotlib.pyplot as pl
import scipy.stats

from numpy.random import seed
from numpy.random import normal

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

N = int(1e4)
dat_mean = 3.4
dat_sig = 2.7
alpha = 0.95

# seed(12)
#dat = normal(loc=0.0, scale=1.0, size=N)
#dat = normal(loc=8.4, scale=6.7, size=N)
dat = normal(loc=dat_mean, scale=dat_sig, size=N)

# pl.figure(); pl.hist(dat, bins=20) 
# pl.show()
# pl.grid(1)

#h = 0.5                  # Interval length
#h = 1.0                   # Interval length
h = 0.25                 # Interval length
#h = 1/16                 # Interval length
lx = np.floor(dat.min()) # Left interval limit
rx = np.ceil(dat.max())  # Right interval limit
nedges = int((rx - lx)/h) + 1
edges = np.linspace(lx, rx, nedges)    # Bin edges
x = edges[:-1] + 0.5*h                 # Bin middle points
# x = h*(0.5 + np.arange(lx, rx))  # Bin middle points
frq, binedg = np.histogram(dat, bins=edges)
nfrq = len(frq)

# raise SystemExit

xmean = np.sum(x*frq)/N
xsig = np.sqrt(np.sum((x**2)*frq)/N - xmean**2)
z = (x - xmean)/xsig  # Standardized x
normal_PDF = np.exp(-z**2/2)/np.sqrt(2*np.pi) # Standard normal PDF
tfrq = (h*N/xsig)*normal_PDF  # Theoretical frequency
chi2 = np.sum((frq - tfrq)**2/tfrq)

# raise SystemExit
    
#
# Group the leftmost and rightmost bins with frequencies less then or equal 5
#
iq1 = int(nfrq/4)    # First quartile
iq4 = int(nfrq*3/4)  # Fourth quartile

j = 0
while j < iq1:
    if frq[j] > 5:
        il = j
        break
    j = j + 1
    print(j, frq[j])

il = il - 1
    
j = nfrq - 1
while j > iq4:
    if frq[j] > 5:
        ir = j
        break
    j = j - 1
    print(j, frq[j])

ir = ir + 1

frql = np.sum(frq[:il+1])
tfrql = np.sum(tfrq[:il+1])
frqr = np.sum(frq[ir:])
tfrqr = np.sum(tfrq[ir:])

print()
print("il = %d, ir = %d" % (il, ir))
print("frql = %g, tfrql = %g" % (frql, tfrql))
print("frqr = %g, tfrqr = %g" % (frqr, tfrqr))

frq_mod = np.copy(frq[il:ir+1])
tfrq_mod = np.copy(tfrq[il:ir+1])

frq_mod[0] = frql
frq_mod[-1] = frqr
tfrq_mod[0] = tfrql
tfrq_mod[-1] = tfrqr

nfrq_mod = len(frq_mod)

chi2_mod = np.sum((frq_mod - tfrq_mod)**2/tfrq_mod)


#
# Critical value of chi2(k) with k degrees of freedom, for significance 0.95
# dat is normal (at significance alpha) if chi2 < chi2cr
#
k = nfrq - 2 - 1
chi2cr = scipy.stats.chi2.ppf(alpha, k)

k_mod = nfrq_mod - 2 - 1
chi2cr_mod = scipy.stats.chi2.ppf(alpha, k_mod)

cmps_mod = ' < ' if chi2cr_mod < chi2cr_mod else ' > '
norm_or_not_mod = ' Normal (mod) ' if chi2_mod < chi2cr_mod \
    else ' NOT Normal (mod)'

print()
print('chi2_mod = %6.2f' % chi2_mod, ', chi2cr_mod = %6.2f' % chi2cr_mod, \
      ' at k_mod = %d degrees of freedom' % k_mod)
print('chi2_mod' + cmps_mod + 'chi2cr_mod: ' + \
      norm_or_not_mod + ' at %4.2f significance level' % alpha)

print()


cmps = ' < ' if chi2 < chi2cr else ' > '
norm_or_not = ' Normal ' if chi2 < chi2cr else ' NOT Normal '


pl.figure();
pl.plot(x, frq, '.', color='blue', label='Observed'); pl.grid(1)
pl.hist(dat, bins=edges, alpha=0.2, color='green', histtype='stepfilled')
# pl.plot(x, frq)
pl.plot(x, tfrq, color='red', label='Theoretical')
pl.legend(fontsize=15, loc='upper right')
pl.title('Pearson Normality test: $N_x$ = %d, $\emph{bin}$ = %g' % \
         (N, h), fontsize=15)
pl.xlabel('$x$', fontsize=18)
pl.ylabel('Frequency', fontsize=15)

pl.figtext(0.14, 0.82, r'$\chi^2_{obs} =$ %6.2f' % chi2, fontsize=15)
pl.figtext(0.14, 0.76, r'$\chi^2_{crit} =$ %6.2f' % chi2cr, fontsize=15)
pl.figtext(0.14, 0.70, r'$\chi^2_{obs} %s \chi^2_{crit}:$' % cmps, fontsize=15)
pl.figtext(0.14, 0.63, r'%s' % norm_or_not, color='red', fontsize=20)
pl.figtext(0.14, 0.58, r'significance %4.2f' % alpha, fontsize=15)
pl.figtext(0.14, 0.52, r'deg. of freedom %d' % k, fontsize=15)

pl.figtext(0.7, 0.63, r'Mean = %5.2f' % dat_mean, fontsize=15)
pl.figtext(0.7, 0.58, r'$\sigma=%5.2f$' % dat_sig, fontsize=15)
pl.figtext(0.7, 0.52, r'$x \in [%d  ..  %d]$' % (lx, rx), fontsize=15)

pl.savefig('fig/Pearson_test_N%d_bin%g_mean%g_std%g.svg' % \
           (N, h, dat_mean, dat_sig), format='svg')

# pl.show()

print('N = ', N, ', nfrq = ', nfrq)
print('x \in [%d .. %d], bin size = %6.4f' % (lx, rx, h))
print('chi2_observed = %6.2f' % chi2, ', chi2_critical = %6.2f' % chi2cr, \
      ' at k = %d degrees of freedom' % k)
print('chi2_observed' + cmps + 'chi2_critical: ' + \
      norm_or_not + ' at %4.2f significance level' % alpha)

pl.show()


