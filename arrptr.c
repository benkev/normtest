#include <stdio.h>
#include <stdlib.h>
// #include <stdint.h>
// #include <unistd.h>
#include <math.h>
// #include <time.h>

int Nfrm = 3, Nch = 6, Nqua = 4;

int main() {
    int Ntot = Nfrm*Nch*Nqua;
    int ifrm, ich, iqua, iseq, num = 0;
    float (*qua)[Nch][Nqua] = malloc(sizeof(float[Nfrm][Nch][Nqua]));
    float *pqua = (float *)qua;
    float q, *pq;
    int idx;
    
    for (ifrm=0; ifrm<Nfrm; ifrm++)
        for (ich=0; ich<Nch; ich++)
            for (iqua=0; iqua<Nqua; iqua++)
                qua[ifrm][ich][iqua] = 100*ifrm + 10*ich + iqua;

    pq = (float *) qua[1];
    for (iseq=0; iseq<Nch*Nqua; iseq++)
        *pq++ = 0.0;

    for (iseq=0; iseq<Ntot; iseq++) {
        printf("%3d %03g\n", iseq, *pqua++);
    }

    pqua = (float *)qua;
    
    /* for (ifrm=0; ifrm<Nfrm; ifrm++) */
    /*     for (ich=0; ich<Nch; ich++) */
    /*         for (iqua=0; iqua<Nqua; iqua++) { */
    /*             idx = (ifrm*Nch + ich)*Nqua + iqua; */
    /*             q = pqua[idx]; */
    /*             printf("idx = %4d, q = %4g\n", idx, q); */

    /*         } */

    /* printf("pqua = %p, qua = %p\n", pqua, qua); */

    /* float *p = (float *) qua[1]; */
    
    /* printf("p = %f\n", p[0]); */
}
