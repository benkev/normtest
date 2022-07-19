#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void reduce_count(__global const float *x, 
                           __global int *sums, 
                           __local int *localSums, 
                           float lo, float hi, int numBins, int n)
{
    uint lid = get_local_id(0);
    uint gid = get_global_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);

    if (lid < numBins){
        localSums[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float step = (hi - lo)/numBins;   /* compute interval step */

    if(gid < n){
        int index = (int) ((x[gid] - lo) / step);   /* compure the interval number */
        if (index == numBins){
            index -= 1;
        }
        atomic_inc(&localSums[index]);   /* increase the local count for interval 'index' */
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < numBins){
        atomic_add(&sums[lid], localSums[lid]);
    }
}