import math
import numpy as np
import matplotlib.pyplot as pl
#from matplotlib.font_manager import FontProperties
# import scipy.stats

# fname_m5b = 'rd1910_wz_268-1811.m5b'
fname_m5b = 'rd1903_ft_100-0950.m5b'

F = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))

pl.rcParams['text.usetex'] = True # Use LaTeX in Matplotlib text

nfrm = 1

ndat = 2500*nfrm   # Total data (32-bit words)
# thr = 0.6652475842498528 # 0.82       # Threshold in STD 
thr = 0.817       # Threshold in STD 
# thr = 0.798121

d = np.zeros(ndat, dtype=np.uint32)   # Raw data
xt = np.zeros_like(d, dtype=np.float64)
x = np.zeros(2500, dtype=np.float64)
qua = np.zeros(4, dtype=np.uint32)  # Quantiles

for ifrm in range(nfrm):
    foff = 10016*ifrm
    i0 = ifrm*2500
    i1 = i0 + 2500
    h = np.fromfile(fname_m5b, dtype=np.uint32, \
                    offset=foff, count=4)
#    print('i0=%d, i1=%d' % (i0,i1))
    d[i0:i1] = np.fromfile(fname_m5b, dtype=np.uint32, \
                    offset=foff+16, count=2500)

#    print('Header: 0x%08x  0x%08x  0x%08x  0x%08x' % tuple(h))

    d01 = 0x03 & d[i0:i1]     # 0th channel, bits 0 and 1
    x[np.where(d01 == 3)] =  1.5
    x[np.where(d01 == 2)] =  0.5
    x[np.where(d01 == 1)] = -0.5
    x[np.where(d01 == 0)] =  -1.5
    
    # d23 = 0b1100 & d[i0:i1]   # 1th channel, bits 0 and 1
    # x[np.where(d23 == 3)] =  1.5
    # x[np.where(d23 == 2)] =  0.5
    # x[np.where(d23 == 1)] = -0.5
    # x[np.where(d23 == 0)] =  -1.5
    
    # pl.figure()
    # pl.hist(x, rwidth=0.5, bins=[-3, -2, -1, 0, 1, 2, 3]); pl.grid(1)

d01t = 0x03 & d   # 0th channel, bits 0 and 1

xt[np.where(d01t == 3)] =  1.5
xt[np.where(d01t == 2)] =  0.5
xt[np.where(d01t == 1)] = -0.5
xt[np.where(d01t == 0)] =  -1.5

for idt in range(ndat):
    qua[d01t[idt]] += 1


F = lambda x: 0.5*(1 + math.erf(x/math.sqrt(2)))  # Normal CDF

Fthr = F(-thr)
# hsnor = np.array([F(-thr), F(0)-F(-thr), F(thr) - F(0), 1 - F(0.92)])
hsnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])    # Normal quantilles

hs, be = np.histogram(xt, bins=[-2, -1, 0, 1, 2])
hsrel = hs/ndat

chi2 = sum((hsnor - hsrel)**2)
        
print('Normal:       %5.3f  %5.3f  %5.3f  %5.3f' % tuple(hsnor))
print('Experimental: %5.3f  %5.3f  %5.3f  %5.3f' % tuple(hsrel))
print('Chi2: %8f' % chi2)



#
# Plot integrals
#
xrul = np.linspace(-3., 3., 51)
fnorm = 1/(2*np.pi)*np.exp(-xrul**2/2)

xloc = [-1.5, -0.5, 0.5, 1.5]
xloc0 = [-1.5, -0.5, 0., 0.5, 1.5]
#xloc0 = [-1.5, -1., -0.5, 0., 0.5, 1., 1.5]
pl.figure()
pl.plot(xrul, 1.5*fnorm, 'b-.', lw=0.5)
pl.bar(xloc, hsnor, width=0.5) # , color='b') # , alpha=0.9)
# a = pl.bar(xloc, hsnor, width=0.5) # , color='b') # , alpha=0.9)
pl.bar(xloc, hsrel, width=0.2, color='orange')
y0, y1 = pl.ylim()
pl.ylim(y0, 1.4*y1)
pl.plot([-1,-1], [0, y1], 'k') #, lw=0.7)
pl.plot([1,1], [0, y1], 'k') #, lw=0.7)
pl.plot([0,0], [0, y1], color='k') #, lw=0.7)
pl.plot(xloc, hsrel, 'r.')
pl.xlim(-3.5, 3.5)
pl.grid(1)
# pl.xticks(xloc0, ['$-\infty \sim -v_0$', '$-v_0 \sim 0$', '$0$',
#                   '$0 \sim v_0$', '$v_0 \sim +\infty$'])
pl.xticks(xloc, ['$-\infty \sim -v_0$', '$-v_0 \sim 0$',
                  '$0 \sim v_0$', '$v_0 \sim +\infty$'], fontsize=11)

# a0 = a[0]
# bcol = a0.get_facecolor()  # Pick the default 'blue' color of the hsrel bars

#font0 = FontProperties()
#font = font0.copy()
#font.set_family('monospace')
#cfont = {'fontname':'Courier'}

pl.text(-1.2, 1.015*y1, '$-v_0$') #, fontsize=12)
pl.text(0.95, 1.015*y1, '$v_0$') #, fontsize=12)
pl.text(-0.04, 1.015*y1, '$0$') #, fontsize=12)

# pl.text(-3.4, 1.3*y1, 'Normal:       %5.3f  %5.3f  %5.3f  %5.3f' % \
#         tuple(hsnor), color='b', fontsize=14)
# pl.text(-3.4, 1.2*y1, 'Experimental: %5.3f  %5.3f  %5.3f  %5.3f' % \
#         tuple(hsrel), color='r', fontsize=14)

pl.text(-3.4, 1.3*y1, 'Normal:', color='b', fontsize=14)
pl.text(-3.4, 1.2*y1, 'Experimental:', color='r', fontsize=14)
pl.text(2.1, 1.2*y1, '$\epsilon^2 =$ %8.2e' % chi2, color='r', fontsize=12)

for itx in range(4):
    pl.text(xloc[itx]-0.2, 1.3*y1, '%5.3f' % hsnor[itx], color='b', \
            fontsize=14)
    pl.text(xloc[itx]-0.2, 1.2*y1, '%5.3f' % hsrel[itx], color='r', \
            fontsize=14)


    
pl.text(-3.3, 1.015*y1, '$v_0 = %4.2f\sigma$' % thr, fontsize=14)
pl.text(1.8, 1.015*y1, '%d Frames' % nfrm, fontsize=13)
    
pl.title('M5B Data vs Normal in Quantiles between $-v_0, 0, and +v_0$', \
         fontsize=15)

# pl.savefig('fig/M5B_vs_Normal_in_Quantiles_nfrms_%d_thres_%4.2fsigma.svg' % \
#                (nfrm, thr), format='svg')

pl.show()
