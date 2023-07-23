/*
 *   ker_m5b_gauss_test_chi2.cu
 *
 * CUDA kernel for Normality (Gaussianity) test for M5B files on GPU
 * Single precision floats.
 *
 * Requires:
 * fminbndf(), 1D function minimum search within bounds [a,b].
 *
 */

#include <fminbndf.cu>

__constant__ float sqrt2 = 1.4142135; /* = sqrt(2.0) float 32-bit precision */
__constant__ int frmwords = 2504; /* 32-bit words in a frame with 4-wd header */
__constant__ int frmbytes = 2504*4;
__constant__ int nfdat = 2500;   /* 32-bit words of data in one frame */
__constant__ int nfhead = 4;     /* 32-bit words of header in one frame */
__constant__ int nch = 16;       /* 16 2-bit channels in each 32-bit word */
__constant__ int nqua = 4;       /* 4 quantiles for each channel */
__constant__ int nchqua = 64;    /* = nch*nqua, total of quantiles for 16 chs */
/* Optimization function parameters */
__constant__ float xatol = 1e-4; /* Absolute error */
__constant__ int maxiter = 20;   /* Maximum number of iterations */


__device__ float fminbndf(float (*func)(float x, float *args), float a, float b,
               float *args, float xatol, int maxiter, float *fval, int *niter, 
               int *flag, int disp);

__device__ float f_normcdf(float x) {
    float f = 0.5*(1.0 + erf(x/sqrt2));
    return f;
}


__device__ float residual(float thresh, float *q_obs) {
    /*
     * The function minimized to find the optimal quantization threshold
     *
     * Inputs:
     * thresh: theshold for the quantization like:
     *         -inf, thresh, 0, thresh, +inf
     * q_obs: array of four quantiles of the experimental data
     *          from M5B files.
     * Returns:
     * eps2: sum of squares of the differences between 4 quantiles of 
     *       the standard Gaussian PDF (mean=0, std=1) and 4 quantiles
     *       of a stream from M5B file data. 
     */

    /* q_norm:  quantile of Gaussian in [-inf .. thresh]  [thresh .. +inf] : */
    /* q_norm0: quantile of Gaussian in [thresh .. 0] and [0 .. thresh] : */
    float q_norm = f_normcdf(-thresh);
    float q_norm0 = 0.5 - q_norm;
    float eps2 = pow(q_norm -  q_obs[0], 2) + pow(q_norm0 - q_obs[1], 2) +
                pow(q_norm0 - q_obs[2], 2) + pow(q_norm -  q_obs[3], 2);
    return eps2;
}


__device__ float calc_chi2(float thresh, float *q_obs) {
    /*
     * The function calculates chi^2 to be minimized to find the optimal
     * quantization threshold
     *
     * Inputs:
     * thresh: theshold for the quantization like:
     *         -inf, thresh, 0, thresh, +inf
     * q_obs[5]: array of four quantiles of the experimental (observation) data
     *          from M5B files in its 0th to 3rd elements and the number of
     *          data in one frame, nfdat_fl, in the 4th element. 
     *
     * Returns:
     * chi2: sum of squares of the differences between 4 quantiles of 
     *       the standard Gaussian PDF (mean=0, std=1) and 4 quantiles
     *       of a stream from M5B file observation data divided by the
     *       observation data quantiles:
     *       chi^2 = sum_{i=1}^4((q_norm_i - q_obs_i)/q_obs_i)
     */

    /* q_norm:  quantile of Gaussian in [-inf .. thresh]  [thresh .. +inf] : */
    /* q_norm0: quantile of Gaussian in [thresh .. 0] and [0 .. thresh] : */
    float nfdat_fl = q_obs[4];
    float q_norm = f_normcdf(-thresh);
    float q_norm0 = nfdat_fl*(0.5 - q_norm); /* Theoretical frequency */
    q_norm =        nfdat_fl*q_norm;          /* Theoretical frequency */
    float chi2 = pow(q_norm -  q_obs[0], 2)/q_obs[0] + \
                 pow(q_norm0 - q_obs[1], 2)/q_obs[1] + \
                 pow(q_norm0 - q_obs[2], 2)/q_obs[2] + \
                 pow(q_norm -  q_obs[3], 2)/q_obs[3];
    return chi2;
}


__global__ void m5b_gauss_test(uint *dat, uint *ch_mask,
                           float *quantl, float *chi2,
                           float *thresh, ushort *flag,
                           ushort *niter, uint nfrm) {

    // size_t ifrm = get_global_id(0);  /* Unique m5b frame and thread number */
    // size_t lwi = get_local_id(0); /* Local work-item # within a work-group */
    // size_t nwg = get_num_groups(0); /* Number of work-groups for dim 0 */
    // size_t ngs = get_global_size(0); /* Number of global work-items, dim 0 */
    // size_t nls = get_local_size(0);  /* Number of local work-items, dim 0 */

    int ifrm = blockDim.x*blockIdx.x + threadIdx.x;


    if (ifrm == 0) {
        printf("CUDA block size: %d threads.\n", blockDim.x);
        // printf("Number of threads per block: %d\n", nls);
        // printf("sizeof(uint) = %ld\n", sizeof(uint)); 
    }
    
    if (ifrm > nfrm) return;  // ====== Ignore extra threads ========== >>>
    
    /* 
     * ch_mask[16]:       Channel 2-bit masks
     * quantl[nfrm,16,4]: Quantiles
     * chi2[nfrm,16]:     chi^2
     * thresh[nfrm,16]:   Thresholds
     * flag[nfrm,16]:     Flags
     * niter[nfrm,16]:    Number of fminbnd() iterations
     */ 
    
    uint ch_bits;
    uint idt, ich, iqua, ixdat, iseq;
    // float (*quantl)[16][4]; /* 4 quantiles of  data for 16 channels */
    float *pqua = NULL;
    float q_obs[5] = {0., 0., 0., 0., 0.}; 
    float  *pchi2 = NULL, *pthresh = NULL;
    ushort *pniter = NULL,  *pflag = NULL;

    int nitr = 0;    /* Number of calls to the optimized function residual() */
    float th0 = 0.;  /* Optimal quantization theshold found */
    float res = 0.;  /* The minimal value of chi^2 at the optimal threshold */
    int flg = 0;     /* Optimization flag  */

    float nfdat_fl = (float) nfdat;
    q_obs[4] = nfdat_fl;  /* number of data in one frame */

    /*
     * Here at least the header must be checked if it only contains the 
     * fill-pattern 0x11223344 words signalling the "missing frame".
     *
     * From Mark_5C_Data_Frame_Specification_memo_058.pdf :
     *
     * Under certain circumstances, the Mark 5C writes a 
     * ‘fill-pattern Data Frame’ in place of missing data. 
     * The fill-pattern Data Frame consists of a normal-length Data Frame 
     * completely filled with a user-specified 32-bit pattern. The correlator 
     * recognizes this fill-pattern data to prevent the correlator from 
     * processing the corresponding data segment.
     *
     * Mark 5B fill-pattern Data Frames are generated exactly like Mark 5C,
     * but with a fixed length of 10016 bytes.
     * The Mark 5B fill-pattern word is (0x11223344).
     */
    int ixhdr = ifrm*frmwords;

    

    /*
     * Set pointers to the ifrm-th frame to process in this thread 
     * (or "work item" in the OpenCL terminology.)
     */
    ixdat = ixhdr + nfhead;  /* Index at the 2500-word data block */
    uint *pdat = dat + ixdat;   

    size_t ix_nch = ifrm*nch;  /* Index at the 16-ch section of 1D arrays */
    pchi2 = chi2 + ix_nch;
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

    // printf("ifrm = %ld, ix_nchqua = %4ld, quantl = %14p, pqua = %14p, " \
    //        "dif=%4ld\n",
    //        ifrm, ix_nchqua, (void*)quantl, (void*)pqua,
    //        (pqua-quantl));

    // if (ifrm ==2) {
    //     printf("M5B 2-bit stream masks:\n");
    //     for (ich=0; ich<16; ich++)
    //         printf("ch_mask[%2u] = %10x\n", ich, ch_mask[ich]);
    // }

    for (idt=0; idt<nfdat; idt++) { /* Data 32b-words in frame count */

        for (ich=0; ich<nch; ich++) {
            ch_bits = pdat[idt] & ch_mask[ich]; /* 2-bit-stream value */

            /* Move the 2-bit-stream of the ich-th channel to the
             * rightmost position in the 32-bit word  chbits
             * to get the quantile index iqua from 0,1,2,3 */
            iqua = ch_bits >> (2*ich);
            
            // if (ifrm == 0 && idt == 0)
            //     printf("%ld: iqua = %u\n", ifrm, iqua);

            pqua[ich*nqua+iqua] += 1.0; /* quantl[ifrm][ich][iqua] += 1.0; */

            // if (ifrm == 0 && idt == 0)
            //     for (i=0; i<4; i++)
            //         printf("%g ", quantl[i]);
            // printf("\n");
        }
    }

    // if (ifrm == 0) {
    //     printf("ifrm = %d; quantiles: ", ifrm);
    //     for (i=0; i<4; i++)
    //         printf("%g ", quantl[i]);
    //     printf("\n");
    // }
    
    /* 
     * Finding optimal quantization thresholds and residuals
     */
    for (ich=0; ich<nch; ich++) {

        /* 1D array pqua_ch[i] == quantl[ifrm][ich][i] */
        float *pqua_ch = quantl + (ifrm*nch + ich)*nqua; 
            
        q_obs[0] = *pqua_ch++;
        q_obs[1] = *pqua_ch++;
        q_obs[2] = *pqua_ch++;
        q_obs[3] = *pqua_ch++;

        // q_obs[0] = (quantl[ifrm][ich][0]);
        // q_obs[1] = (quantl[ifrm][ich][1]);
        // q_obs[2] = (quantl[ifrm][ich][2]);
        // q_obs[3] = (quantl[ifrm][ich][3]);


        /*
         * Fit the Gaussian PDF to the quantiles of the signals from 
         * the M5B file. The single variable search Brent's  method is 
         * used find the optimal value of the signal rms (i.e. STD), 
         * which provides the minimum residual between quantiles of 
         * the Gaussian PDF and those of the 2-bit streams 
         * from M5B files. 
         */
        
        th0 = fminbndf(*calc_chi2, 0.5, 1.5, q_obs, xatol, maxiter,
                       &res, &nitr, &flg, 0);
        // if (ifrm < 10) {
        //     printf("KERNEL: ifrm: %d, chi2 = %e\n", ifrm, res);
        // }
        
        pchi2[ich] = res;
        pthresh[ich] = th0;
        pniter[ich] = nitr;
        pflag[ich] = flg;
            
        // chi2[ifrm][ich] = res;
        // thresh[ifrm][ich] = th0;
        // niter[ifrm][ich] = nitr;
        // flag[ifrm][ich] = flg;
            
    }

    return; 
}






