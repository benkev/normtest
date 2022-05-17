import numpy as np

def fminbound(func, bounds, args=(), xatol=1e-5, maxiter=500, disp=0):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.
    xatol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    maxfun = maxiter
    # Test bounds are of correct form
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds

    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    flag = 0
    header = ' Func-count     x          f(x)          Procedure'
    step = '       initial'

    sqrt_eps = np.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = np.inf
    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    if disp > 2:
        print(" ")
        print(header)
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola

            print("Chk accpt parabola: p = %g, q = %g, r = %g, " \
                   "np.abs(0.5*q*r) = %g\na = %g, b = %g, xf = %g" %
                   (p, q, r, np.abs(0.5*q*r), a, b, xf))
            print("q*(a - xf) = %g, q*(b - xf) = %g\n" \
                   "np.abs(p) < np.abs(0.5*q*r) = %s, p > q*(a - xf) = %s, " \
                   "p < q*(b - xf) = %s" %
                  (q*(a - xf), q*(b - xf), str(np.abs(p) < np.abs(0.5*q*r)),
                  str(p > q*(a - xf)), str(p < q*(b - xf))))
            print("(np.abs(p) < np.abs(0.5*q*r)) && (p > q*(a - xf)) " \
                   "&& (p < q*(b - xf)) = %s" % 
                   str((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
                           (p < q*(b - xf))))
             
            if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):

                print("Parabola accepted.");
                
                rat = (p + 0.0) / q
                x = xf + rat
                
                step = '       parabolic'

                print("Parabolic: x = %g, a = %g, b = %g, tol2 = %g\n"  \
                       "(x - a) < tol2 = %s, (b - x) < tol2) = %s" %
                       (x, a, b, tol2, str((x - a) < tol2),
                        str((b - x) < tol2)));
            
                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        print("golden = %d" % golden);
        
        if golden:  # do a golden-section step

            print("Golden accepted.")
            
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

            step = '       golden'

            print("xf = %g, xm = %g, e = %g, rat = %g" % (xf, xm, e, rat))
            
        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)        
        fu = func(x, *args)
        num += 1
        
        fmin_data = (num, x, fu)
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    fval = fx

    if flag == 0: print('Solution found.')
    if flag == 1: print('Maximum number of function calls reached.')
    

    return xf


