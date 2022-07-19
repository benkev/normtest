__global__ void kernel_getHist(unsigned char* array, long size, 
                               unsigned int* histo, int buckets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid>=size)   return;

    unsigned char value = array[tid];

    int bin = value % buckets;

    atomicAdd(&histo[bin],1);
}

void getHist(unsigned char* array, long size, unsigned int* histo,
             int buckets)
{
    unsigned char* dArray;
    cudaMalloc(&dArray,size);
    cudaMemcpy(dArray,array,size,cudaMemcpyHostToDevice);

    unsigned int* dHist;
    cudaMalloc(&dHist,buckets * sizeof(int));
    cudaMemset(dHist,0,buckets * sizeof(int));

    dim3 block(32);
    dim3 grid((size + block.x - 1)/block.x);

    kernel_getHist<<<grid,block>>>(dArray,size,dHist,buckets);

    cudaMemcpy(histo,dHist,buckets * sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(dArray);
    cudaFree(dHist);
}