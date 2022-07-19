__kernel void reduce(__global const float *x, __global float *sums, 
                     __local float *localSums, float a, float b, int n)
{
    uint lid = get_local_id(0);
    uint gid = get_global_id(0);
    uint group_size = get_local_size(0);

    /* is the record in interval [a, b) */
    if(x[gid] >= a && x[gid] < b && gid < n){ 
         localSums[lid] = 1.0;   
    }
    else{
         localSums[lid] = 0.0;
    }

    for (uint stride=group_size/2; stride>0; stride /= 2){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < stride){
            localSums[lid] += localSums[lid + stride];
        }
    }
    if(lid == 0){
        sums[get_group_id(0)] = localSums[lid];
    }
}