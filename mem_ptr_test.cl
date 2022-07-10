__kernel void mem_ptr_test(__global uint *dat) {

    __private float * __private p; /* __private ptr p -->  to _private float */
    __global float * __private g; /* __private ptr g -->  to _global float */
    __global static int gldata[128]; /* 128 integers in global memory */
    __local float *lf; /* __private ptr lf -->  to _local float */
    __global char * __local lgc[8]; /* 8 __local ptrs lgc --> to _global chrs */

}
