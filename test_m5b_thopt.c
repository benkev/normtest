/*
 *   test_m5b_thopt.c
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

float sqrt2 = sqrt(2.0);   /* Global */
float fmega = pow(1024.,2);   /* Global */
float fgiga = pow(1024.,3);   /* Global */
int frmwords = 2504; /* 32-bit words in one frame including the 4-word header */
int frmbytes = 2504*4;
int nfdat = 2500;   /* 32-bit words of data in one frame */

float fminbndf(float (*func)(float x, float *args), float a, float b,
               float *args, float xatol, int maxiter, float *fval, int *niter, 
               int *flag, int disp);

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
     * The function minimized to find the optimal quantization threshold
     *
     * Inputs:
     * thresh: theshold for the quantization like:
     *         -inf, thresh, 0, thresh, +inf
     * q_exprm: array of four quantiles of the experimental data
     *          from M5B files.
     * Returns:
     * chi2: sum of squares of the differences between 4 quantiles of 
     *       the standard Gaussian PDF (mean=0, std=1) and 4 quantiles
     *       of a stream from M5B file data. 
     */

    /* q_norm:  quantile of Gaussian in [-inf .. thresh]  [thresh .. +inf] : */
    /* q_norm0: quantile of Gaussian in [thresh .. 0] and [0 .. thresh] : */
    float q_norm = f_normcdf(-thresh);
    float q_norm0 = 0.5 - q_norm;
    float chi2 = pow(q_norm -  q_exprm[0], 2) + pow(q_norm0 - q_exprm[1], 2) +
                 pow(q_norm0 - q_exprm[2], 2) + pow(q_norm -  q_exprm[3], 2);
    return chi2;
}


        

int main() {
    size_t m5bbytes;
    __uint32_t chbits;
    __uint32_t *dat=NULL;
    __uint32_t ifrm, idt, ich, ifrmdat, iqua, ptim1, ptim2, pdat, i, iseq;
    float (*qua)[16][4]; /* 4 quantiles of the exprm. data for 16 channels */
    float *pqua;
    float q_exprm[4] = {1., 2., 3., 4.};
    float sum_qua = 0.0;
    __uint32_t ch_mask[16];        /*  2-bit masks for all channels */

    
    /* Optimization function parameters */
    float xatol = 1e-4;
    int maxiter = 20;

    /* Results */
    float (*thr)[16];    /* Optimal quantization thesholds found */
    float (*qresd)[16]; /* Residuals between normal and exprm. quantiles */
    int (*flag)[16];   /* Optimization flags  */
    int (*niter)[16]; /* Numbers of calls to residual() function */
    int nitr = 0;    /* Number of calls to the optimized function residual() */
    float res; /* The minimal value of the quantization threshold */
    float th0; /* Optimal quantization theshold found */
    int flg;   /* Optimization flag  */

    
    FILE *fh = NULL;
    FILE *fth = NULL, *fqr = NULL, *ffl = NULL, *fitr = NULL, *fout = NULL;

    int nfrm = 100;
    int nch = 16;   /* 16 2-bit channels in each 32-bit word */
    int nqua = 4;   /* 4 quantiles for each channel */
    int nchqua = nch*nqua; /* Total of quantiles for 16 chans, 64 */
    float nfdat_fl = (float) nfdat;
    int total_frms = 0;
    int last_frmbytes = 0;
    int last_frmwords = 0;

    clock_t tic, toc;
 
    
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
     * Get the m5b file size in m5bbytes
     */
    fseek(fh, 0L, SEEK_END);
    m5bbytes = ftell(fh);
    rewind(fh);

    total_frms = m5bbytes / frmbytes;
    last_frmbytes = m5bbytes % frmbytes;
    last_frmwords = last_frmbytes / 4;
    
    printf("M5B file size: %ld bytes = %g MiB = %g GiB\n",
           m5bbytes, m5bbytes/fmega, m5bbytes/fgiga);
    printf("Frame size: %d Bytes = %d words.\n", frmbytes, nfdat);
    printf("Number of whole frames: %d\n", total_frms);
    printf("Last frame size: %d Bytes = %d words.\n",
           last_frmbytes, last_frmwords);
    
    /* nfrm = total_frms; // Uncomment to read in the TOTAL M5B FILE */
    

    /* dat = (__uint32_t *) malloc(m5bbytes*sizeof(__uint32_t)); */

    dat = (__uint32_t *) malloc(frmwords*nfrm*sizeof(__uint32_t));

    /* 
     * Read the whole m5b file into RAM 
     */
    /* fread(dat, sizeof(__uint32_t), m5bbytes, fh); */

    time(&tic);

    /* 
     * Read nfrm frames from m5b file into RAM 
     */
    fread(dat, sizeof(__uint32_t), frmwords*nfrm, fh);

    fclose(fh);

    time(&toc);

    printf("M5B file has been read. Time: %ld s.\n", toc - tic);

    time(&tic); /* Start computations */
    
    qua =   malloc(sizeof(float[nfrm][16][4]));
    pqua = (float *)qua;
    qresd = malloc(sizeof(float[nfrm][16]));
    thr =   malloc(sizeof(float[nfrm][16]));
    flag =  malloc(sizeof(int[nfrm][16]));
    niter =  malloc(sizeof(int[nfrm][16]));
                               
    ptim1 = 2; /* Pointer to the word with whole seconds in header */
    ptim2 = 3; /* Pointer to the word with tenths of milliseconds in header */
    pdat = 4;  /* Pointer to the 2500-word data block */
    
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */

    /*     Print time within a second from frame header */
    /*     printf("%05x", dat[ptim1] & 0xfffff);   // Whole seconds of time */
    /*     printf(".%04x\n", dat[ptim2] / 0x10000); // Tenths of milliseconds */

        /* Zeroize the quantiles for current frame: all channels. */
        pqua = (float *) qua[ifrm];
        for (iseq=0; iseq<nchqua; iseq++)
            *pqua++ = 0.0;
        /* for (ich=0; ich<16; ich++) */
        /*     for (iqua=0; iqua<4; iqua++) */
        /*         qua[ifrm][ich][iqua] = 0.0; */


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

            q_exprm[0] = (qua[ifrm][ich][0]) / nfdat_fl;
            q_exprm[1] = (qua[ifrm][ich][1]) / nfdat_fl;
            q_exprm[2] = (qua[ifrm][ich][2]) / nfdat_fl;
            q_exprm[3] = (qua[ifrm][ich][3]) / nfdat_fl;


            /*
             * Fit the Gaussian PDF to the quantiles of the signals from 
             * the M5B file. The single variable search Brent's  method is 
             * used find the optimal value of the signal rms (i.e. STD), 
             * which provides the minimum residual between quantiles of 
             * the Gaussian PDF and those of the 2-bit streams 
             * from M5B files. 
             */
            th0 = fminbndf(*residual, 0.5, 1.5, q_exprm, xatol, 20,
                           &res, &nitr, &flg, 0);

            qresd[ifrm][ich] = res;
            thr[ifrm][ich] = th0;
            niter[ifrm][ich] = nitr;
            flag[ifrm][ich] = flg;
        }
        
        /* Move pointers to the next frame */
        pdat = pdat + frmwords;
        ptim1 = ptim1 + frmwords;
        ptim2 = ptim2 + frmwords;
        
    }                 /* for (ifrm=0 ... */

    time(&toc); /* End computations */
    
    long tictoc = (long) (toc - tic);
    long tmin = tictoc/60, tsec = tictoc%60;
    if (tmin == 0) 
        printf("Computations took time: %ld s.\n", tictoc);
    else
        printf("Computations took time: %ld min. %ld s.\n", tmin, tsec);


    fth = fopen("thresh.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        /* fprintf(fth, "%d: ", ifrm); */
        /* printf("%d: ", ifrm); */
        for (ich=0; ich<16; ich++) { /* Channel count */
            fprintf(fth, "%8g ", thr[ifrm][ich]);
            /* printf("%8g ", thr[ifrm][ich]); */
        }
        fprintf(fth, "\n");
        /* printf("\n"); */
    }
    fclose(fth);

    fqr = fopen("qresd.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        /* fprintf(fqr, "%d: ", ifrm); */
        /* printf("%d: ", ifrm); */
        for (ich=0; ich<16; ich++) { /* Channel count */
            fprintf(fqr, "%12g ", qresd[ifrm][ich]);
            /* printf("%10g ", qresd[ifrm][ich]); */
        }
        fprintf(fqr, "\n");
        /* printf("\n"); */
    }
    fclose(fqr);

    fitr = fopen("niter.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        /* fprintf(fitr, "%d: ", ifrm); */
        /* printf("%d: ", ifrm); */
        for (ich=0; ich<16; ich++) { /* Channel count */
            fprintf(fitr, "%2d ", niter[ifrm][ich]);
            /* printf("%2d ", niter[ifrm][ich]); */
        }
        fprintf(fitr, "\n");
        /* printf("\n"); */
    }
    fclose(fitr);

    ffl = fopen("flags.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        /* fprintf(ffl, "%d: ", ifrm); */
        /* printf("%d: ", ifrm); */
        for (ich=0; ich<16; ich++) { /* Channel count */
            fprintf(ffl, "%1d ", flag[ifrm][ich]);
            /* printf("%1d ", flag[ifrm][ich]); */
        }
        fprintf(ffl, "\n");
        /* printf("\n"); */
    }
    fclose(ffl);


    int chn = 4;
    
    fout = fopen("quantiles_4chn.txt","w");
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */
        for (iqua=0; iqua<4; iqua++) {
            fprintf(fout, "%g ", qua[ifrm][chn][iqua]);
            /* printf("%g ", qua[ifrm][chn][iqua]); */
        }
        fprintf(fout, "\n");
        /* printf("\n"); */
    }
    fclose(fout);

    free(dat);
}






