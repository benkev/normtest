#include <stdio.h>
#include <stdlib.h>
// #include <stdint.h>
// #include <unistd.h>
#include <math.h>
// #include <time.h>

int Nfrm = 4, Nch = 6, Nqua = 4, Ndat = 2500;

int main() {
    int Ntot = Nfrm*Nch*Nqua;
    int ifrm, ich, iqua, iseq, num = 0;
    int *dat = malloc(sizeof(float[Ndat]));
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

    /* Test how &dat[M]+N works */
    printf("dat = %08p, &dat[16] = %08p, &dat[16]+4 = %08p\n",
           dat, &dat[16], &dat[16]+4);
    printf("dat = %d, &dat[16] = %d, &dat[16]+4 = %d\n",
           dat, &dat[16], &dat[16]+4);

    ifrm = 2;
    pqua = (float *)&qua[ifrm];
    for (ich=0; ich<Nch; ich++)
        for (iqua=0; iqua<Nqua; iqua++)
            printf("%03g\n", pqua[ich*Nqua+iqua]);


    

    /* pqua = (float *)qua; */
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


    /* Even though array and &array both are resulting in same address
     * but they are different types of addresses. And this is the
     * difference between array and &array.
     *
     * Basically, array is a pointer to the first element of array but &array 
     * is a pointer to whole array of 5 int. Since array is pointer to int, 
     * addition of 1 resulted in an address with increment of 4 (assuming int 
     * size in your machine is 4 bytes). Since &array is pointer to array of 5 
     * ints, addition of 1 resulted in an address with increment of 
     * 4 x 5 = 20 = 0x14. Now you see why these two seemingly similar pointers 
     * are different at core level. 
     *
     * This logic can be extended to multidimensional arrays as well. Suppose 
     *   double array2d[5][4];
     * is a 2D array. Here, array2d is a pointer to array of 4 int but 
     * &array2d is pointer to array of 5 rows arrays of 4 int. If this sounds
     * cryptic, you can always have a small program to print these after 
     * adding 1.
     *
     * We hope that we could clarify that any array name itself is a pointer to
     * the first element but & (i.e. address-of) for the array name is a pointer
     * to the whole array itself. */
    
    int array[5];
    double array2d[5][4];
    
    /* If %p is new to you, you can use %d as well */
    printf("array=%08p : &array=%08p\n", array, &array); 
    printf("array+1 = %08p : &array+1 = %08p\n", array+1, &array+1);
    
    printf("array2d=%08p : &array2d=%08p\n", array2d, &array2d); 
    printf("array2d+1 = %08p : &array2d+1 = %08p\n", array2d+1, &array2d+1);
    
}
