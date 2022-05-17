/*
 * test_fminf.c
 *
 * Single precision version
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

float fminbndf(float (*func)(float x, float *args), float a, float b,
                float *args, float xatol, int maxiter, float *fval, int *flag,
                int disp);
float fun(float x, float *args);

float fun(float x, float *args) {
    float f;
    f = (x-2.0)*(x+1.0)*x;
    return f;
}

int main(int argc, char *argv[]) {
    float x0 = 0.0;
    float fval = 0.0;
    int flag = 0;
    float a, b, xatol = 1e-3;
    float args[2] = {7., 9.};

    /* printf ("argc = %d\n", argc); */

    switch (argc) {
    case 4:
        xatol = strtof(argv[3], NULL);
    case 3: {
        a = strtof(argv[1], NULL);
        b = strtof(argv[2], NULL);
        break;
    }
    default:
        printf("At least two arguments, a and b, a < b, needed, " \
               "and (optional) xatol.\n");
        return 1;
        // break;
    }

    /* a = atof(argv[1]); */
    /* b = atof(argv[2]); */
    
    
    x0 = fminbndf(*fun, a, b, args, xatol, 50, &fval, &flag, 3);

    printf("x0 = %8f, fval = %8f, flag = %d\n", x0, fval, flag);
    
    return 0;
}





