/*
 * test_fmin.c
 *
 * Double precision version
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

double fminbnd(double (*func)(double x, double *args), double a, double b,
                 double *args, double xatol, int maxiter, double *fval,
                 int *flag, int disp);
double fun(double x, double *args);

double fun(double x, double *args) {
    double f = 0.0;
    f = (x-2.0)*(x+1.0)*x;
    /* printf("args[0] = %g, args[1] = %g\n", args[0], args[1]); */
    return f;
}

int main(int argc, char *argv[]) {
    double x0 = 0.0;
    double fval = 0.0;
    int flag = 0;
    double a, b, xatol = 1e-3;
    double args[2] = {7., 9.};

    /* printf ("argc = %d\n", argc); */

    switch (argc) {
    case 4:
        xatol = strtof(argv[3], NULL);
    case 3: {
        a = strtof(argv[1], NULL);
        b = strtof(argv[2], NULL);
        printf("test_fmin: a=%g, b=%g, xatol=%g\n", a, b, xatol);
        break;
    }
    default:
        printf("At least two arguments, a and b, a < b, needed, " \
               "and (optional) xatol.\n");
        return 1;
    }

    /* a = atof(argv[1]); */
    /* b = atof(argv[2]); */

    
    x0 = fminbnd(*fun, a, b, args, xatol, 15, &fval, &flag, 3);

    printf("x0 = %8f, fval = %8f, flag = %d\n", x0, fval, flag);
    
    return 0;
}





