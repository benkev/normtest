/*
 *   test_m5b_thopt.c
 *
 */
// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <unistd.h>
// #include <math.h>
// #include <time.h>

#define A(x,y) a[x*width + y]   // ????????????????????????????????????????

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


// prg.gausstestm5b(queue, (Nproc,), (Nwitem,), buf_dat, buf_quantl, buf_residl,
//              buf_thresh, buf_flag, buf_niter, nfrm)
        

// int main() {

__kernel void gausstestm5b(__global int *dat, __global uint *ch_mask,
                           __global float *quantl, __global float *residl,
                           __global float *thresh, __global short *flag,
                           __global short *niter, uint nfrm) {
    size_t m5bbytes;
    uint chbits;
    uint ifrm, idt, ich, iqua, ptim1, ptim2, pdat, i;
    float (*quantl)[16][4]; /* 4 quantiles of the exprm. data for 16 channels */
    float *pquantl;
    float q_exprm[4] = {1., 2., 3., 4.};
    float sum_qua = 0.0;
    // uint ch_mask[16];        /*  2-bit masks for all channels */

    
    /* Optimization function parameters */
    float xatol = 1e-4;
    int maxiter = 20;

    /* Results */
    // float (*thresh)[16];    /* Optimal quantization thesholds found */
    // float (*qresd)[16]; /* Residuals between normal and exprm. quantiles */
    // int (*flag)[16];   /* Optimization flags  */
    // int (*niter)[16]; /* Numbers of calls to residual() function */

    int nitr = 0;    /* Number of calls to the optimized function residual() */
    float res; /* The minimal value of the quantization threshold */
    float th0; /* Optimal quantization theshold found */
    int flg;   /* Optimization flag  */

    float nfdat_fl = (float) nfdat;

    // /*
    //  * Create 16 2-bit masks for 16 channels
    //  */
    // ch_mask[0] = 0x00000003;       /* Mask for channel 0 */
    // for (ich=1; ich<16; ich++)
    //     ch_mask[ich] = ch_mask[ich-1] << 2;
    
    // /* for (ich=0; ich<16; ich++) */
    // /*     printf("ch_mask[%2d] = %08x = %032b\n", ich, ch_mask[ich]); */

    ptim1 = 2; /* Pointer to the word with whole seconds in header */
    ptim2 = 3; /* Pointer to the word with tenths of milliseconds in header */
    pdat = 4;  /* Pointer to the 2500-word data block */
    
    for (ifrm=0; ifrm<nfrm; ifrm++) { /* Frame count */

        /* Zeroize the quantiles for current frame: all channels. */
        for (ich=0; ich<16; ich++)
            for (iqua=0; iqua<4; iqua++)
                quantl[ifrm][ich][iqua] = 0.0;

        for (idt=0; idt<nfdat; idt++) /* Data 32b-words in frame count */
            for (ich=0; ich<16; ich++) {
                chbits = dat[pdat+idt] & ch_mask[ich]; /* 2-bit-stream value */
                /* Move the 2-bit-stream of the ich-th channel to the
                 * rightmost position in the 32-bit word 
                 * to get the quantile index from 0,1,2,3 */
                iqua = chbits >> 2*ich; 
                quantl[ifrm][ich][iqua] += 1.0;
            }


        /* 
         * Finding optimal quantization thresholds and residuals
         */
        for (ich=0; ich<16; ich++) {
            /*
             * Normalize the quantiles dividing them by the frame size
             * so that their sum be 1: sum(q_exprm) == 1.
             */

            q_exprm[0] = (quantl[ifrm][ich][0]) / nfdat_fl;
            q_exprm[1] = (quantl[ifrm][ich][1]) / nfdat_fl;
            q_exprm[2] = (quantl[ifrm][ich][2]) / nfdat_fl;
            q_exprm[3] = (quantl[ifrm][ich][3]) / nfdat_fl;


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
            thresh[ifrm][ich] = th0;
            niter[ifrm][ich] = nitr;
            flag[ifrm][ich] = flg;
        }
        
        /* Move pointers to the next frame */
        pdat = pdat + frmwords;
        ptim1 = ptim1 + frmwords;
        ptim2 = ptim2 + frmwords;
        
    }                 /* for (ifrm=0 ... */

}






