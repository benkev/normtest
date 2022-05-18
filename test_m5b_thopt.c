/*
    test m5b.c

 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

/* Set channel values to channel 4 */
#define ch00 0
#define ch01 1 << 8
#define ch10 2 << 8
#define ch11 3 << 8

float sqrt2 = sqrt(2.0);   /* Global */
float fmega = pow(1024.,2);   /* Global */
float fgiga = pow(1024.,3);   /* Global */


float fminbndf(float (*func)(float x, float *args), float a, float b,
                float *args, float xatol, int maxiter, float *fval, int *flag,
                int disp);
float f_normcdf(float x) {
    float f = 0.5*(1.0 + erf(x/sqrt2));
    return f;
}

int main() {
    size_t sz;
    __uint32_t chbits;
    __uint32_t *dat=NULL;
    size_t ifrm, idt, ich, ifrmdat, iqua, ptim1, ptim2, pdat, i;
    long (*qua)[16][4]; /* 4 quantiles of the experiment data for 16 channels */
    long *pqua;
    long sum_qua = 0;
    __uint32_t ch_mask[16];        /*  2-bit masks for all channels */

    FILE *fh = NULL;

    /* int vector[5] = {1, 2, 3, 4, 5}; */
    /* int *pv = vector; */

    int nfrm = 3;
    
    qua = malloc(sizeof(long[nfrm][16][4]));
    pqua = (long *)qua;


    /*
     * Create 16 2-bit masks for 16 channels
     */
    ch_mask[0] = 0x00000003;       /* Mask for channel 0 */
    for (ich=1; ich<16; ich++)
        ch_mask[ich] = ch_mask[ich-1] << 2;
    
    /* for (ich=0; ich<16; ich++) */
    /*     printf("ch_mask[%2d] = %08x = %032b\n", ich, ch_mask[ich]); */

    fh = fopen("rd1910_wz_268-1811.m5b","rb");
    /*
     * Get the m5b file size in sz
     */
    fseek(fh, 0L, SEEK_END);
    sz = ftell(fh);
    rewind(fh);

    printf("M5B file size: %ld B = %g MiB = %g GiB\n", sz, sz/fmega, sz/fgiga);

    return 0;
    
    /* dat = (__uint32_t *) malloc(sz*sizeof(__uint32_t)); */

    dat = (__uint32_t *) malloc(2504*nfrm*sizeof(__uint32_t));

    /* 
     * Read the whole m5b file into RAM 
     */
    /* fread(dat, sizeof(__uint32_t), sz, fh); */

    /* 
     * Read nfrm frames from m5b file into RAM 
     */
    fread(dat, sizeof(__uint32_t), 2504*nfrm, fh);


    
    ptim1 = 2; /* Pointer to the word with whole seconds in header */
    ptim2 = 3; /* Pointer to the word with tenths of milliseconds in header */
    pdat = 4;  /* Pointer to the 2500-word data block */
    
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        
    /*     Print time within a second from frame header */
    /*     printf("%05x", dat[ptim1] & 0xfffff);   // Whole seconds of time */
    /*     printf(".%04x\n", dat[ptim2] / 0x10000); // Tenths of milliseconds */

        /* Zeroize the quantiles for current frame: all channels. */
        for (ich=0; ich<16; ich++)
            for (iqua=0; iqua<4; iqua++)
                qua[ifrm][ich][iqua] = 0;

        for (idt=0; idt<2500; idt++) /* Data 32b-words in ifrm-th frame count */
            for (ich=0; ich<16; ich++) {
                chbits = dat[pdat+idt] & ch_mask[ich]; /* 2-bit-stream value */
                /* Move the 2-bit-stream of the ich-th channel to the
                 * rightmost position in the 32-bit word 
                 * to get the quantile index from 0,1,2,3 */
                iqua = chbits >> 2*ich; 
                qua[ifrm][ich][iqua]++;
            }
    /*     sum_qua = 0; */
    /*     for (iqua=0; iqua<4; iqua++) */
    /*         sum_qua += qua[iqua];    /\* Sum up all the four quantiles *\/ */
        
    /*     printf("quantiles: "); */
    /*     for (iqua=0; iqua<4; iqua++) */
    /*         printf("%ld ", qua[iqua]); */
    /*     printf("  sum: %ld", sum_qua); */
    /*     printf("\n"); */

    /*     for (iqua=0; iqua<4; iqua++) qua[iqua] = 0; */
            
        pdat = pdat + 2504;
        ptim1 = ptim1 + 2504;
        ptim2 = ptim2 + 2504;
    }

    
    
   
    fclose(fh);

    // sleep(15);

    free(dat);
}






