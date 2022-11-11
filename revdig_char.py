import numpy as np
import sys

x0 = np.int16(sys.argv[1])

s = np.int16(-1) if x0 < 0 else np.int16(1)
x = s * x0

sy = []
isy = 0

i10 = np.int16(10)
y = np.int16(0)

while x:
    q = x // i10
    r = x % i10
    sy.append(str(r))
    isy = isy + 1
    print("x=%6d; q=%6d; r=%6d; sy=%6s" % (x, q, r, sy))
    x = q
    print("x=%6d; q=%6d" % (x, q))

    
print("x=%6d" % x0)
print("y=%6s" % sy)

