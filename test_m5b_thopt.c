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
int frlen = 2504;
int nfdat = 2500;


float fminbndf(float (*func)(float x, float *args), float a, float b,
                float *args, float xatol, int maxiter, float *fval, int *flag,
                int disp, FILE *fd);

float f_normcdf(float x) {
    float f = 0.5*(1.0 + erf(x/sqrt2));
    return f;
}

/* def quanerr(thr, args): */
/*     Fthr = F(-thr) */
/*     hsnor = np.array([Fthr, 0.5-Fthr, 0.5-Fthr, Fthr])  # Normal quantiles */
/*     hsexp = np.copy(args) */
/*     err = sum((hsnor - hsexp)**2) */
/*     return err */


float residual(float thresh, float *q_exprm) {
    /*
     * thresh: theshold for the quantization like:
     *         -inf, thresh, 0, thresh, +inf
     * q_exprm: array of four quantiles of the experimental data
     * from M5B files.
     */

    /* To accelerate, we do not put the Gaussian quantiles into the
     * array qua_gauss[4], but use them immediately in the error calculation. */
    /* static float */
    /*     qua_gauss[4] = {q_norm, 0.5-q_norm, 0.5-q_norm, q_norm}; */
    /* Quantile of Gaussian in [-inf .. thresh]  [thresh .. +inf] : */
    float q_norm = f_normcdf(-thresh);
    /* Quantile of Gaussian in [thresh .. 0] and [0 .. thresh] : */
    float q_norm0 = 0.5 - q_norm;
    /* To accelerate, we do not use for loop to find the sum of four squares */
    float err = pow(q_norm - q_exprm[0], 2) + pow(q_norm0 - q_exprm[0], 2) +
                pow(q_norm - q_exprm[0], 2) + pow(q_norm0 - q_exprm[0], 2);
    /* printf("residual: q_exprm = %g %g %g %g\n", q_exprm[0], q_exprm[1], */
    /*        q_exprm[2], q_exprm[3]);  */
    return err;
}


        

int main() {
    size_t sz;
    __uint32_t chbits;
    __uint32_t *dat=NULL;
    __uint32_t ifrm, idt, ich, ifrmdat, iqua, ptim1, ptim2, pdat, i;
    float (*qua)[16][4]; /* 4 quantiles of the exprm. data for 16 channels */
    float *pqua;
    float q_exprm[4] = {1., 2., 3., 4.};
    float sum_qua = 0.0;
    __uint32_t ch_mask[16];        /*  2-bit masks for all channels */

    
    /* Optimization function parameters */
    float xatol = 1e-4;
    int maxiter = 20;
    int flg;
    float res;
    float th0;

    /* Results */
    float (*thr)[16];  /* Optimal quantization thesholds found */
    float (*qresd)[16]; /* Residuals between normal and exprm. quantiles */
    int (*flag)[16];   /* Optimization flags  */
    

    
    FILE *fh = NULL;
    FILE *fth = NULL, *fqr = NULL, *ffl = NULL, *fout = NULL;

    int nfrm = 1000;
    float fl_nfdat = (float) nfdat;

    
    qua =   malloc(sizeof(float[nfrm][16][4]));
    pqua = (float *)qua;
    qresd = malloc(sizeof(float[nfrm][16]));
    thr =   malloc(sizeof(float[nfrm][16]));
    flag =  malloc(sizeof(int[nfrm][16]));
                               


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

    fclose(fh);
    
    printf("M5B file has been read.\n");


    /*
     * Open fminbndf() diagnostic file
     */
    fh = fopen("fminbndf.txt","w");
 

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
                qua[ifrm][ich][iqua] = 0.0;

        for (idt=0; idt<nfdat; idt++) /* Data 32b-words in frame count */
            for (ich=0; ich<16; ich++) {
                chbits = dat[pdat+idt] & ch_mask[ich]; /* 2-bit-stream value */
                /* Move the 2-bit-stream of the ich-th channel to the
                 * rightmost position in the 32-bit word 
                 * to get the quantile index from 0,1,2,3 */
                iqua = chbits >> 2*ich; 
                qua[ifrm][ich][iqua] += 1.0;
            }


        /* 
         * Finding optimal quantization thresholds and residuals
         */
        for (ich=0; ich<16; ich++) {
            /*
             * Normalize the quantiles dividing them by the frame size
             * so that their sum be 1: sum(q_exprm) == 1.
             */

            q_exprm[0] = (qua[ifrm][ich][0]) / fl_nfdat;
            q_exprm[1] = (qua[ifrm][ich][1]) / fl_nfdat;
            q_exprm[2] = (qua[ifrm][ich][2]) / fl_nfdat;
            q_exprm[3] = (qua[ifrm][ich][3]) / fl_nfdat;


            /*
             * Fit the Gaussian PDF to the quantiles of the signals from 
             * the M5B file. The single variable search Brent's  method is 
             * used find the optimal value of the signal rms (i.e. STD), 
             * which provides the minimum residual between quantiles of 
             * the Gaussian PDF and those of the 2-bit streams 
             * from M5B files. 
             */
            th0 = fminbndf(*residual, 0.5, 1.5, q_exprm, xatol, 20,
                           &res, &flg, 0, fh);

            qresd[ifrm][ich] = res;
            thr[ifrm][ich] = th0;
            flag[ifrm][ich] = flg;
        }
        
        /* Move pointers to the next frame */
        pdat = pdat + frlen;
        ptim1 = ptim1 + frlen;
        ptim2 = ptim2 + frlen;
        
    }                 /* for (ifrm=0 ... */

    fclose(fh);

    
    fth = fopen("thresh.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        fprintf(fth, "%d: ", ifrm);
        /* printf("%d: ", ifrm); */
        for (ich=0; ich<16; ich++) { /* Channel count */
            fprintf(fth, "%g ", thr[ifrm][ich]);
            /* printf("%g ", thr[ifrm][ich]); */
        }
        fprintf(fth, "\n");
        /* printf("\n"); */
    }
    fclose(fth);

    fqr = fopen("qresd.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        fprintf(fqr, "%d: ", ifrm);
        /* printf("%d: ", ifrm); */
        for (ich=0; ich<16; ich++) { /* Channel count */
            fprintf(fqr, "%g ", qresd[ifrm][ich]);
            /* printf("%g ", qresd[ifrm][ich]); */
        }
        fprintf(fqr, "\n");
        /* printf("\n"); */
    }
    fclose(fqr);

    ffl = fopen("flags.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        fprintf(ffl, "%d: ", ifrm);
        /* printf("%d: ", ifrm); */
        for (ich=0; ich<16; ich++) { /* Channel count */
            fprintf(ffl, "%d ", flag[ifrm][ich]);
            /* printf("%d ", flag[ifrm][ich]); */
        }
        fprintf(ffl, "\n");
        /* printf("\n"); */
    }
    fclose(ffl);

   
    fout = fopen("quantiles.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        for (iqua=0; iqua<4; iqua++) {
            fprintf(fout, "%g ", qua[ifrm][4][iqua]);
            /* printf("%g ", qua[ifrm][4][iqua]); */
        }
        fprintf(fout, "\n");
        /* printf("\n"); */
    }
    fclose(fout);

    free(dat);
}






