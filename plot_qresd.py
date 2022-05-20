import numpy as np
import matplotlib.pyplot as pl

q = np.loadtxt('qresd.txt')
r = np.reshape(q[:16*16,:], (64,64))

pl.figure(); pl.pcolormesh(r); pl.colorbar(shrink=0.8)
pl.title('$\chi^2$ of Difference bw Normal and M5B Quantiles')
pl.xlabel('4 Columns 16 Channels Each')
pl.ylabel('$\chi^2$ for 64 Words')

pl.show()


