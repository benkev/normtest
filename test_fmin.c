/*
 * test_fmin.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

float fminbound(float (*func)(float x, float *args), float a, float b,
                float *args, float xatol, int maxiter, int disp);

float fun(float x, float *args) {
    float f;
    f = (x-2)*(x+1)*x;
    return f;
}

int main() {
    float x0 = 0.0;

    x0 = fminbound(fun, 0.0, 2.0, NULL, 1e-5, 500, 3);

    printf("x0 = %f8", x0);
    
    return 0;
}





