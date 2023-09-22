#import math
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
from numpy import sqrt, exp, pi

# pl.rcParams['text.usetex'] = True
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    })

pl.ion()

npdf = lambda x: (1/sqrt(2*pi))*exp(-0.5*x**2)
# F = lambda x: 0.5*(1 + erf(x/sqrt(2)))
 
xrul = np.linspace(-3, 3, 501)

fx = npdf(xrul)
th = 1.2

fig = pl.figure()
pl.plot(xrul, fx, 'g', lw=2.5)
pl.plot((-3,3), (0,0), 'k')
pl.plot((-th,-th), (0, 1.05*npdf(-th)), 'k-.')
pl.plot([0,0], [0, 1.02*npdf(0)], 'k-.')
pl.plot([th,th], [0, 1.05*npdf(th)], 'k-.')

# pl.fill_between(xrul, npdf(xrul), where=(-3.3<xrul)&(xrul<=-th),
#                 color='b', alpha=0.2)
# pl.fill_between(xrul, npdf(xrul), where=(-th<xrul)&(xrul<=0),
#                 color='m', alpha=0.2);
# pl.fill_between(xrul, npdf(xrul), where=(0<xrul)&(xrul<=th),
#                 color='g', alpha=0.2)
# pl.fill_between(xrul, npdf(xrul), where=((th<xrul)&(xrul<10)),
#                 color='c', alpha=0.2)

# pl.text(-th-0.8, 0.03, r"$\Phi(-\theta)$", fontsize=14)
# pl.text(-th+0.06, 0.03, r"$\frac{1}{2} - \Phi(-\theta)$", fontsize=14)
# pl.text(0.06, 0.03, r"$\frac{1}{2} - \Phi(-\theta)$", fontsize=14)
# pl.text(th+0.15, 0.03, r"$\Phi(-\theta)$", fontsize=14)

pl.text(-th-0.55, 0.03, r"$E_0$", fontsize=14)
pl.text(-th+0.5, 0.03, r"$E_1$", fontsize=14)
pl.text(0.45, 0.03, r"$E_2$", fontsize=14)
pl.text(th+0.25, 0.03, r"$E_3$", fontsize=14)

pl.text(0.8, 0.36, r"$E_0 = E_3 = N \Phi(-\theta)$", fontsize=14)
pl.text(0.8, 0.33, r"$E_1 = E_2 = N\left[\frac{1}{2} - \Phi(-\theta)\right]$",
        fontsize=14)

pl.axis("off")
pl.tight_layout(pad=1.5)

# pl.xticks([-th, 0, th], [r'$-\theta$', r'$0$', r'$\theta$'], fontsize=16)
# pl.yticks([])
xh = 0.02
pl.figtext(0.30, xh, r'$-\theta$', fontsize=16)
pl.figtext(0.49, xh, r'$0$', fontsize=16)
pl.figtext(0.66, xh, r'$\theta$', fontsize=16)

fig.savefig("fig_npdf_areas.eps", format="eps", dpi=1200, bbox_inches="tight",
            transparent=True)


pl.show()

