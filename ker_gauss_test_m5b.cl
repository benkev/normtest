/*
 *   test_m5b_thopt.c
 *
 */
// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <unistd.h>
#include <math.h>
// #include <time.h>

#define A(x,y) a[x*width + y]   // ????????????????????????????????????????

__constant float sqrt2 = sqrt(2.0);   /* Global */
__constant int frmwords = 2504; /* 32-bit words in a frame with 4-word header */
__constant int frmbytes = 2504*4;
__constant int nfdat = 2500;   /* 32-bit words of data in one frame */
__constant int nfhead = 4;   /* 32-bit words of header in one frame */
__constant int nch = 16;   /* 16 2-bit channels in each 32-bit word */
__constant int nqua = 4;   /* 4 quantiles for each channel */
__constant int nchqua = nch*nqua; /* Total of quantiles for 16 chans, 64 */

float fminbndf(float (*func)(float x, float *args), float a, float b,
               float *args, float xatol, int maxiter, float *fval, int *niter, 
               int *flag, int disp);

float f_normcdf(float x) {
    float f = 0.5*(1.0 + erf(x/sqrt2));
    return f;
}


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

__kernel void gausstestm5b(__global uint *dat, __global uint *ch_mask,
                           __global float *quantl, __global float *residl,
                           __global float *thresh, __global short *flag,
                           __global short *niter, uint nfrm) {

    size_t ifrm = get_global_id(0);  /* Unique process and frame number */
    
    uint ch_bits;
    uint ifrm, idt, ich, iqua, ixtim1, ixtim2, ixdat, i, iseq;
    // float (*quantl)[16][4]; /* 4 quantiles of  data for 16 channels */
    float *pqua;
    float q_exprm[4];
    
    /* Optimization function parameters */
    float xatol = 1e-4;
    int maxiter = 20;

    int nitr = 0;    /* Number of calls to the optimized function residual() */
    float res; /* The minimal value of the quantization threshold */
    float th0; /* Optimal quantization theshold found */
    int flg;   /* Optimization flag  */

    float nfdat_fl = (float) nfdat;

    /*
     * Set pointers to the ifrm-th frame to process in this thread 
     * (or "work item" in the OpenCL terminology.)
     */
    ixdat = ifrm*frmwords + nfhead;  /* Index at the 2500-word data block */
    uint *pdat = dat + ixdat;   
    
    float *pqresd = (float *) qresd[ifrm];
    float *pthr =   (float *) thr[ifrm];
    short *pniter = (short *) niter[ifrm];
    short *pflag =  (short *) flag[ifrm];


    // /*
    //  * Create 16 2-bit masks for 16 channels
    //  */
    // ch_mask[0] = 0x00000003;       /* Mask for channel 0 */
    // for (ich=1; ich<16; ich++)
    //     ch_mask[ich] = ch_mask[ich-1] << 2;
    
    
    /* Zeroize the quantiles for current frame: all channels. */
    pqua = (float *) quantl[Nchqua*ifrm];
    for (iseq=0; iseq<Nchqua; iseq++)
        *pqua++ = 0.0;
    
     /* 1D array pqua[i] == qua[ifrm][i] */
    pqua = (float *) &quantl[Nchqua*ifrm];
    
    for (idt=0; idt<nfdat; idt++) /* Data 32b-words in frame count */
        for (ich=0; ich<nch; ich++) {
            ch_bits = pdat[idt] & ch_mask[ich]; /* 2-bit-stream value */
            /* Move the 2-bit-stream of the ich-th channel to the
             * rightmost position in the 32-bit word  chbits
             * to get the quantile index iqua from 0,1,2,3 */
            iqua = ch_bits >> (2*ich);
            pqua[ich*nqua+iqua] += 1.0; /* quantl[ifrm][ich][iqua] += 1.0; */
        }


    /* 
     * Finding optimal quantization thresholds and residuals
     */
    for (ich=0; ich<nch; ich++) {
        /*
         * Normalize the quantiles dividing them by the frame size
         * so that their sum be 1: sum(q_exprm) == 1.
         */

        /* 1D array pqua_ch[i] == quantl[ifrm][ich][i] */
        pqua_ch = (float *) quantl[(ifrm*nch + ich)*nqua]; // ???????????????
            
        q_exprm[0] = *pqua_ch++ / nfdat_fl;
        q_exprm[1] = *pqua_ch++ / nfdat_fl;
        q_exprm[2] = *pqua_ch++ / nfdat_fl;
        q_exprm[3] = *pqua_ch++ / nfdat_fl;

        // q_exprm[0] = (quantl[ifrm][ich][0]) / nfdat_fl;
        // q_exprm[1] = (quantl[ifrm][ich][1]) / nfdat_fl;
        // q_exprm[2] = (quantl[ifrm][ich][2]) / nfdat_fl;
        // q_exprm[3] = (quantl[ifrm][ich][3]) / nfdat_fl;


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

        pqresd[ich] = res;
        pthr[ich] = th0;
        pniter[ich] = nitr;
        pflag[ich] = flg;
            
        // qresd[ifrm][ich] = res;
        // thresh[ifrm][ich] = th0;
        // niter[ifrm][ich] = nitr;
        // flag[ifrm][ich] = flg;
            
    }
    return; 
}






