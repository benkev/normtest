import numpy as np
import sys

x0 = np.int16(sys.argv[1])

s = np.int16(-1) if x0 < 0 else np.int16(1)
x = s * x0

i10 = np.int16(10)
y = np.int16(0)
j = 12
while x and j:
    q = x // i10
    r = x % i10
    i10y = i10*y
    #print("i10=%6d; y=%6d; i10y=%6d" % (i10, y, i10y))
    if y != 0 and i10y / y != i10:
        print("Overflow! 10*%d = %d > 32767." % (y, i10y))
        y = 0
        break
    if y != 0 and r > i10y:
        print("Overflow! 10*%d + %d > 32767." % (y, r))
        y = 0
        break
    y = i10*y + r
    print("x=%6d; q=%6d; r=%6d; y=%6d" % (x, q, r, y))
    x = q
    j = j - 1

    
print("0x=%6d" % x0)
print("y=%6d" % y)

