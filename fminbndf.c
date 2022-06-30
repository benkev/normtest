/*
 * fminbndf.c
 *
 * Single precision version
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <float.h>

#define sign(x) ((x) > 0.0) ? 1.0 : (((x) < 0.0) ? -1.0 : 0.0)
/* #define min(X, Y)  ((X) < (Y) ? (X) : (Y)) */
/* #define max(X, Y)  ((X) < (Y) ? (X) : (Y)) */
#define strbool(b) ((b) ? "True" : "False")


float fminbndf(float (*func)(float x, float *args), float a, float b,
               float *args, float xatol, int maxiter, float *fval, int *niter,
               int *flag, int disp) {

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
    /* 
     *     Returns
     *     xf : float
     *         The func argument that provides its minimum within
     *         the specified interval [a, b].
     *         NAN in case of failure.
     *     fval : float
     *         The minimum value of func() within the specified
     *         interval [a, b], i.e. func(xf). 
     *     flag : int
     *         Flag indicating optimization success or failure.
     *         flag == 0: Solution found.
     *         flag == 1: Maximum number of function calls reached.
     *         flag == 2: Error: Wrong bounds: a > b.
     */

    int golden, si; 
    float fx, fu, ffulc, fnfc, xm, tol1, tol2;
    float p, q, r;
    const float sqrt_eps = sqrt(FLT_EPSILON); // sqrt(1.19209e-07);
    /* const float sqrt_eps = sqrt(DBL_EPSILON); // sqrt(2.22044604925e-16); */
    const float golden_mean = 0.5*(3.0 - sqrt(5.0));
    float fulc = 0;
    float nfc = fulc;
    float xf = fulc;
    float rat = 0.0;
    float e = 0.0;
    float x = xf;
    //    float f = 0.0;

    int num = 1;
    
    char *header = " Func-count     x          f(x)          Procedure\n";
    char *step = "       initial\n";

    fx = (*func)(x, args);

    if (a > b) {
        printf("The lower bound exceeds the upper bound: a > b.\n");
        *flag = 2;
        *fval = NAN;
        return NAN;
    }
    
    *fval = 0.0; /* Assume solution found */
    *flag = 0;
    *niter = 0;
    
    fulc = a + golden_mean*(b - a);
    nfc = xf = fulc;
    rat = e = 0.0;
    x = xf;
    fx = (*func)(x, args);
    fu = INFINITY;
    ffulc = fnfc = fx;
    xm = 0.5*(a + b);
    tol1 = sqrt_eps*fabs(xf) + xatol/3.0;
    tol2 = 2.0*tol1;

    if (disp > 2) {
        printf("\n");
        printf("%s", header);
        printf("%5d   %12.6g %12.6g %s", 1, xf, fx, step);
    }
    
    while (fabs(xf - xm) > (tol2 - 0.5*(b - a))) {
        golden = 1;
        /* Check for parabolic fit */
        if (fabs(e) > tol1) {
            golden = 0;
            r = (xf - nfc) * (fx - ffulc);
            q = (xf - fulc) * (fx - fnfc);
            p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if (q > 0.0)
                p = -p;
            q = fabs(q);
            r = e;
            e = rat;

            /* Check for acceptability of parabola */

            if ((fabs(p) < fabs(0.5*q*r)) && (p > q*(a - xf)) &&
                (p < q*(b - xf))) {
                rat = (p + 0.0)/q;
                x = xf + rat;

                step = "       parabolic\n";
            
                if (((x - a) < tol2) || ((b - x) < tol2)) {
                    si = sign(xm - xf) + ((xm - xf) == 0.0);
                    rat = tol1 * si;
                }
            }
            else      /* do a golden-section step */
                golden = 1;
        }

        if (golden) {  /* do a golden-section step */
            
            if (xf >= xm)
                e = a - xf;
            else
                e = b - xf;
            rat = golden_mean*e;

            step = "       golden\n";
        }
        
        si = sign(rat) + (rat == 0.0);
        x = xf + si*fmax(fabs(rat), tol1);
        fu = (*func)(x, args);
        num += 1;
       
        if (disp > 2)
            printf("%5d   %12.6g %12.6g %s", num, x, fu, step);

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
        tol1 = sqrt_eps * fabs(xf) + xatol/3.0;
        tol2 = 2.0*tol1;

        if (num >= maxiter) {
            *flag = 1;
            break;
        }
    }   /* while(fabs(xf - xm) ... */
    
    if (isnan(xf) || isnan(fx) || isnan(fu))
        *flag = 2;

    *fval = fu;
    *niter = num;

    return xf;
}

