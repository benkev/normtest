#
# call_fminbound.py
#
import numpy as np
from fminbound import fminbound

x = np.linspace(-2.1,2.1,101)

y = lambda x: (x-2)*(x+1)*x

x0 = fminbound(y, [1.2, 1.23], xatol=1e-5,  disp=3)

print('x0 = %7f' % x0)


