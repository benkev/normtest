/*
 * fminbound.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

#define sign(x) ((x) > 0) ? 1 : ((x < 0) ? -1 : 0)
/* #define min(X, Y)  ((X) < (Y) ? (X) : (Y)) */
/* #define max(X, Y)  ((X) < (Y) ? (X) : (Y)) */

float fminbound(float (*func)(), float a, float b, float *args, float xatol,
                int maxiter, int disp) {

    /* def fminbound(func, bounds, args=(), xatol=1e-5, maxiter=500, disp=0): */
    /*     """ */
    /*     Options */
    /*     ------- */
    /*     maxiter : int */
    /*         Maximum number of iterations to perform. */
    /*     disp: int, optional */
    /*         If non-zero, print messages. */
    /*             0 : no message printing. */
    /*             1 : non-convergence notification messages only. */
    /*             2 : print a message on convergence too. */
    /*             3 : print iteration results. */
    /*     xatol : float */
    /*         Absolute error in solution `xopt` acceptable for convergence. */

    /*     """ */

    float fulc, nfc, xf, rat, e, x, fx, fu, ffulc, fnfc, xm, tol1, tol2;
    float p, q, r, fval;
    int golden, num, si; 
    int maxfun = maxiter;
    int flag = 0;
    float sqrt_eps = sqrt(2.2e-16);
    float golden_mean = 0.5*(3.0 - sqrt(5.0));

    char *header = " Func-count     x          f(x)          Procedure";
    char *step = "       initial";

    if (a > b) {
        printf("The lower bound exceeds the upper bound.");
        return 0;
    }
    
    fulc = a + golden_mean*(b - a);
    nfc = xf = fulc;
    
    rat = e = 0.0;
    x = xf;
    fx = func(x, *args);
    num = 1;
    /*    fmin_data = (1, xf, fx)  */
    fu = INFINITY;

    ffulc = fnfc = fx;
    xm = 0.5 * (a + b);
    tol1 = sqrt_eps * fabsf(xf) + xatol/3.0;
    tol2 = 2.0 * tol1;

    if (disp > 2) {
        printf("\n");
        printf("%s", header);
        printf("%5.0f   %12.6g %12.6g %s", 1.0, xf, fx, step);
    }
    
    while (fabsf(xf - xm) > (tol2 - 0.5*(b - a))) {
        golden = 1;
        /* Check for parabolic fit */
        if (fabsf(e) > tol1) {
            golden = 0;
            r = (xf - nfc) * (fx - ffulc);
            q = (xf - fulc) * (fx - fnfc);
            p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if (q > 0.0)
                p = -p;
            q = fabsf(q);
            r = e;
            e = rat;

            /* Check for acceptability of parabola */
            if ((fabsf(p) < fabsf(0.5*q*r)) && (p > q*(a - xf)) &&
                (p < q*(b - xf))) { 
                rat = (p + 0.0)/q;
                x = xf + rat;

                step = "       parabolic";
                
                if (((x - a) < tol2) || ((b - x) < tol2)) {
                    si = sign(xm - xf) + ((xm - xf) == 0);
                    rat = tol1 * si;
                }
                else      // do a golden-section step
                    golden = 1;
            }
        }
        if (golden)  // do a golden-section step
            if (xf >= xm)
                e = a - xf;
            else {
                e = b - xf;
                rat = golden_mean*e;

                step = "       golden";
            }
        
        si = sign(rat) + (rat == 0);
        x = xf + si*fmax(fabsf(rat), tol1);
        fu = func(x, *args);
        num += 1;
                    
        if (disp > 2)
            printf("%5.0f   %12.6g %12.6g %s", num, x, fu, step);

        if (fu <= fx) {
            if (x >= xf)
                a = xf;
            else
                b = xf;
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        }
        else {
            if (x < xf)
                a = x;
            else
                b = x;
            if ((fu <= fnfc) || (nfc == xf)) {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            }
            else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * fabsf(xf) + xatol/3.0;
        tol2 = 2.0*tol1;

        if (num >= maxfun) {
            flag = 1;
            break;
        }
    }
    if (isnan(xf) || isnan(fx) || isnan(fu))
        flag = 2;

    fval = fx;

    /* 
     * flag == 0: Solution found.
     * flag == 1: Maximum number of function calls reached.
     */

    return xf;
}

