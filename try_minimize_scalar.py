import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize
from scipy.optimize import minimize_scalar

pl.ion()

def f(x):
    return (x - 2) * x * (x + 2)**2

# scipy.optimize.minimize_scalar(
#     fun,
#     bracket=None,
#     bounds=None,
#     args=(),
#     method='brent',
#     tol=None,
#     options=None,
# )

res1 = minimize_scalar(f)
print("res1 = ", res1)

res2 = minimize_scalar(f, bounds=(-3, -1), method='bounded')
print("res2 = ", res2)

x = np.linspace(-3, 3, 101)
pl.figure()
pl.plot(x, f(x)); pl.grid(1)
pl.plot(res1["x"], res1["fun"], "ro")
pl.plot(res2["x"], res2["fun"], "ro")

pl.show()

