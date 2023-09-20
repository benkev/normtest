#import math
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
from numpy import sqrt, exp, pi

pl.rcParams['text.usetex'] = True
pl.ion()

npdf = lambda x: (1/sqrt(2*pi))*exp(-0.5*x**2)
# F = lambda x: 0.5*(1 + erf(x/sqrt(2)))
 
xrul = np.linspace(-3, 3, 501)

fx = npdf(xrul)
th = 1.2

pl.figure()
pl.plot(xrul, fx, 'g')
pl.plot((-3,3), (0,0), 'k')
pl.plot((-th,-th), (0, 1.05*npdf(-th)), 'k-.')
pl.plot([0,0], [0, 1.02*npdf(0)], 'k-.')
pl.plot([th,th], [0, 1.05*npdf(th)], 'k-.')

pl.fill_between(xrul, npdf(xrul), where=(-3.3<xrul)&(xrul<=-th),
                color='b', alpha=0.2)
pl.fill_between(xrul, npdf(xrul), where=(-th<xrul)&(xrul<=0),
                color='m', alpha=0.2);
pl.fill_between(xrul, npdf(xrul), where=(0<xrul)&(xrul<=th),
                color='g', alpha=0.2)
pl.fill_between(xrul, npdf(xrul), where=((th<xrul)&(xrul<10)),
                color='c', alpha=0.2)

pl.text(-th-0.8, 0.03, r"$\Phi(-\theta)$", fontsize=14)
pl.text(-th+0.06, 0.03, r"$\frac{1}{2} - \Phi(-\theta)$", fontsize=14)
pl.text(0.06, 0.03, r"$\frac{1}{2} - \Phi(-\theta)$", fontsize=14)
pl.text(th+0.15, 0.03, r"$\Phi(-\theta)$", fontsize=14)

pl.xticks([-th, 0, th], [r'$-\theta$', r'$0$', r'$\theta$'], fontsize=16)
pl.yticks([])


pl.show()

