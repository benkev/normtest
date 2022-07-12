/*
 * devquery_ocl.c
 *
 */
#include <CL/cl.h>
#include <stdio.h>
#include <math.h>

const float fkilo =       1024.0;
const float fmega =    1048576.0;  /* = pow(1024.,2) */
const float fgiga = 1073741824.0;  /* = pow(1024.,3) */



int main() {
	cl_platform_id platform;
    cl_uint nentrs = 1, ndevs;
	cl_int err;
    cl_ulong size_g, size_l, size_p;

	clGetPlatformIDs(1, &platform, NULL);

    /* Find number of GPUs in ndevs */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &ndevs);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "ERROR clGetDeviceIDs(), err = %d\n", err);
        return 0;
	}
    printf("Number of available GPUs: %u\n", ndevs);


    /* Get array of all the available GPUs in devs */
    cl_device_id devs[ndevs];
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, nentrs, devs, NULL);

    /* Get info on the 0-th GPU */
    cl_device_id devID = devs[0];
    clGetDeviceInfo(devID, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(cl_ulong), &size_g, 0);
    clGetDeviceInfo(devID, CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(cl_ulong), &size_l, 0);

    
    printf("CL_DEVICE_LOCAL_MEM_SIZE = %uB, %.2fKB, %.2fMB, %.2fGB\n", size_g,
           (float)size_g/fkilo, (float)size_g/fmega, (float)size_g/fgiga);

    printf("CL_DEVICE_LOCAL_MEM_SIZE = %uB, %.2fKB\n", size_l,
           (float)size_l/fkilo);

    return 0;
}
