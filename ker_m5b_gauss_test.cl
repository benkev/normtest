/*
 *   ker_m5b_gauss_test.cl
 *
 */

#include <fminbndf.cl>

__constant float sqrt2 = 1.4142135; /* = sqrt(2.0) float 32-bit precision */
__constant int frmwords = 2504; /* 32-bit words in a frame with 4-word header */
__constant int frmbytes = 2504*4;
__constant int nfdat = 2500;   /* 32-bit words of data in one frame */
__constant int nfhead = 4;     /* 32-bit words of header in one frame */
__constant int nch = 16;       /* 16 2-bit channels in each 32-bit word */
__constant int nqua = 4;       /* 4 quantiles for each channel */
__constant int nchqua = 64;    /* = nch*nqua, total of quantiles for 16 chans */

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


// prg.m5b_gauss_test(queue, (Nproc,), (Nwitem,), buf_dat, buf_quantl, buf_residl,
//              buf_thresh, buf_flag, buf_niter, nfrm)
        

// int main() {

__kernel void m5b_gauss_test(__global uint *dat, __global uint *ch_mask,
                           __global float *quantl, __global float *residl,
                           __global float *thresh, __global ushort *flag,
                           __global ushort *niter, uint nfrm) {

    size_t ifrm = get_global_id(0);  /* Unique m5b frame and thread number */

    printf("proc#: %ld\n", ifrm);

    if (ifrm > nfrm) return;  // ======== Ignore extra threads============ >>>
    
    /* 
     * ch_mask[16]:       Channel 2-bit masks
     * quantl[nfrm,16,4]: Quantiles
     * residl[nfrm,16]:   Residuals
     * thresh[nfrm,16]:   Thresholds
     * flag[nfrm,16]:     Flags
     * niter[nfrm,16]:    Number of fminbnd() iterations
     */ 
    
    uint ch_bits;
    uint idt, ich, iqua, ixtim1, ixtim2, ixdat, i, iseq;
    // float (*quantl)[16][4]; /* 4 quantiles of  data for 16 channels */
    float *pqua = NULL, *pqua_ch = NULL;
    float q_exprm[4];
    float  *presidl = NULL, *pthresh = NULL;
    ushort *pniter = NULL,  *pflag = NULL;


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

    size_t ix_nch = ifrm*nch;  /* Index at the 16-ch section of 1D arrays */
    presidl = residl + ix_nch;
    pthresh = thresh + ix_nch;
    pniter = niter + ix_nch;
    pflag =  flag + ix_nch;


    /* 
     * Zeroize the quantiles for current frame: all channels. 
     */
    size_t ix_nchqua = ifrm*nchqua;  /* Index at the 16x4 section of quantl */

    pqua = quantl + ix_nchqua; /* 1D array pqua[i] == quantl[ifrm][i] */
    for (iseq=0; iseq<nchqua; iseq++)
        *pqua++ = 0.0;
    /*
     * Sum up the quantiles from the 2500 data words for all 16 channels
     */
    pqua = quantl + ix_nchqua; /* 1D array pqua[i] == quantl[ifrm][i] */
    
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
        float *pqua_ch = quantl + (ifrm*nch + ich)*nqua; 
            
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

        presidl[ich] = res;
        pthresh[ich] = th0;
        pniter[ich] = nitr;
        pflag[ich] = flg;
            
        // residl[ifrm][ich] = res;
        // thresh[ifrm][ich] = th0;
        // niter[ifrm][ich] = nitr;
        // flag[ifrm][ich] = flg;
            
    }
    return; 
}






