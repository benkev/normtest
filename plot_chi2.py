help_txt = '''
plot_chi2.py
'''

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as pl

df = 3

x = np.linspace(chi2.ppf(0.001, df), chi2.ppf(0.999, df), 100)
x095 = np.linspace(0.95, chi2.ppf(0.999, df), 20)


pl.grid(1)
pl.plot(x, chi2.pdf(x, df), 'r', label='chi2 pdf')
pl.fill(x095, chi2.pdf(x095, df), 'm')

pl.plot([-0.2,x[-1]], [0,0], 'k')    # X axis
pl.plot([0,0], [-0.005, 0.25], 'k')  # Y axis

pl.show()

