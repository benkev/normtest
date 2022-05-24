import numpy as np
import matplotlib.pyplot as pl

q = np.loadtxt('qresd.txt')
r = np.reshape(q[:16*16,:], (64,64))

pl.figure(figsize=(9,7))
# pl.pcolormesh(r, cmap=pl.cm.hot)
pl.pcolormesh(r, cmap=pl.cm.jet)
# pl.pcolormesh(r)
pl.colorbar(shrink=0.8)
pl.title('$\chi^2$ of Difference bw Normal and M5B Quantiles')
pl.xlabel('4 Columns 16 Channels Each')
pl.ylabel('$\chi^2$ for 64 Frames')

pl.plot([16,16], [0,64], 'w', lw=0.5)
pl.plot([32,32], [0,64], 'w', lw=0.5)
pl.plot([48,48], [0,64], 'w', lw=0.5)

# pl.plot([0,64], [48,48], 'w', lw=0.5)
# pl.plot([0,64], [32,32], 'w', lw=0.5)
# pl.plot([0,64], [16,16], 'w', lw=0.5)


th = np.loadtxt('thresh.txt')
rh = np.reshape(th[:16*16,:], (64,64))

pl.figure(figsize=(9,7))
# pl.pcolormesh(rh, cmap=pl.cm.hot)
pl.pcolormesh(rh, cmap=pl.cm.jet)
# pl.pcolormesh(rh)
pl.colorbar(shrink=0.8)
pl.title('Optimal Quantization Thresholds')
pl.xlabel('4 Columns 16 Channels Each')
pl.ylabel('Q-Thresholds for 64 Frames')

pl.plot([16,16], [0,64], 'w', lw=0.5)
pl.plot([32,32], [0,64], 'w', lw=0.5)
pl.plot([48,48], [0,64], 'w', lw=0.5)

# pl.plot([0,64], [48,48], 'w', lw=0.5)
# pl.plot([0,64], [32,32], 'w', lw=0.5)
# pl.plot([0,64], [16,16], 'w', lw=0.5)


pl.show()


