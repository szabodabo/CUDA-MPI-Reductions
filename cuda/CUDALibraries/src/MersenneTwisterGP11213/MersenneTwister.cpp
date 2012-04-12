/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates the use of CURAND to generate
 * random numbers on GPU and CPU.  
 */

// Utilities and system includes
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Utilities and system includes
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrQATest.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime_api.h>
#include <curand.h>

float compareResults(int rand_n, float *h_RandGPU, float *h_RandCPU);

const int    DEFAULT_RAND_N = 2400000;
const unsigned int DEFAULT_SEED = 777;

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if( cudaSuccess != err) {
		    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCurandErrors(err)           __checkCurandErrors (err, __FILE__, __LINE__)

    inline void __checkCurandErrors( curandStatus_t err, const char *file, const int line )
    {
        if( CURAND_STATUS_SUCCESS != err) {
            fprintf(stderr, "%s(%i) : checkCurandErrors() CURAND error %d: ", file, line, (int)err);
            switch (err) {
                case CURAND_STATUS_VERSION_MISMATCH:    fprintf(stderr, "CURAND_STATUS_VERSION_MISMATCH");
                case CURAND_STATUS_NOT_INITIALIZED:     fprintf(stderr, "CURAND_STATUS_NOT_INITIALIZED");
                case CURAND_STATUS_ALLOCATION_FAILED:   fprintf(stderr, "CURAND_STATUS_ALLOCATION_FAILED");
                case CURAND_STATUS_TYPE_ERROR:          fprintf(stderr, "CURAND_STATUS_TYPE_ERROR");
                case CURAND_STATUS_OUT_OF_RANGE:        fprintf(stderr, "CURAND_STATUS_OUT_OF_RANGE"); 
                case CURAND_STATUS_LENGTH_NOT_MULTIPLE: fprintf(stderr, "CURAND_STATUS_LENGTH_NOT_MULTIPLE");
                case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: 
				                fprintf(stderr, "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED");
                case CURAND_STATUS_LAUNCH_FAILURE:      fprintf(stderr, "CURAND_STATUS_LAUNCH_FAILURE"); 
                case CURAND_STATUS_PREEXISTING_FAILURE: fprintf(stderr, "CURAND_STATUS_PREEXISTING_FAILURE");
                case CURAND_STATUS_INITIALIZATION_FAILED:     
				                fprintf(stderr, "CURAND_STATUS_INITIALIZATION_FAILED");
                case CURAND_STATUS_ARCH_MISMATCH:       fprintf(stderr, "CURAND_STATUS_ARCH_MISMATCH");
                case CURAND_STATUS_INTERNAL_ERROR:      fprintf(stderr, "CURAND_STATUS_INTERNAL_ERROR");
                default: fprintf(stderr, "CURAND Unknown error code\n");
            }
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // General GPU Device CUDA Initialization
    int gpuDeviceInit(int devID)
    {
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }
        if (devID < 0) 
            devID = 0;
        if (devID > deviceCount-1) {
            fprintf(stderr, "\n");
            fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
            fprintf(stderr, "\n");
            return -devID;
        }

        cudaDeviceProp deviceProp;
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        if (deviceProp.major < 1) {
            fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
            exit(-1);                                                  \
        }

        checkCudaErrors( cudaSetDevice(devID) );
        printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
        return devID;
    }

    // This function returns the best GPU (with maximum GFLOPS)
    int gpuGetMaxGflopsDeviceId()
    {
	    int current_device   = 0, sm_per_multiproc = 0;
	    int max_compute_perf = 0, max_perf_device  = 0;
	    int device_count     = 0, best_SM_arch     = 0;
	    cudaDeviceProp deviceProp;

	    cudaGetDeviceCount( &device_count );
	    // Find the best major SM Architecture GPU device
	    while ( current_device < device_count ) {
		    cudaGetDeviceProperties( &deviceProp, current_device );
		    if (deviceProp.major > 0 && deviceProp.major < 9999) {
			    best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		    }
		    current_device++;
	    }

        // Find the best CUDA capable GPU device
        current_device = 0;
        while( current_device < device_count ) {
           cudaGetDeviceProperties( &deviceProp, current_device );
           if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
               sm_per_multiproc = 1;
		   } else {
               sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
           }

           int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
           if( compute_perf  > max_compute_perf ) {
               // If we find GPU with SM major > 2, search only these
               if ( best_SM_arch > 2 ) {
                   // If our device==dest_SM_arch, choose this, or else pass
                   if (deviceProp.major == best_SM_arch) {	
                       max_compute_perf  = compute_perf;
                       max_perf_device   = current_device;
                   }
               } else {
                   max_compute_perf  = compute_perf;
                   max_perf_device   = current_device;
               }
           }
           ++current_device;
	    }
	    return max_perf_device;
    }

    // Initialization code to find the best CUDA Device
    int findCudaDevice(int argc, const char **argv)
    {
        cudaDeviceProp deviceProp;
        int devID = 0;
        // If the command-line has a device number specified, use it
        if (checkCmdLineFlag(argc, argv, "device")) {
            devID = getCmdLineArgumentInt(argc, argv, "device=");
            if (devID < 0) {
                printf("Invalid command line parameters\n");
                exit(-1);
            } else {
                devID = gpuDeviceInit(devID);
                if (devID < 0) {
                   printf("exiting...\n");
                   shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                   exit(-1);
                }
            }
        } else {
            // Otherwise pick the device with highest Gflops/s
            devID = gpuGetMaxGflopsDeviceId();
            checkCudaErrors( cudaSetDevice( devID ) );
            checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
            printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
        }
        return devID;
    }
// end of CUDA Helper Functions

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Start logs
    shrQAStart(argc, argv);

    // initialize the GPU, either identified by --device
    // or by picking the device with highest flop rate.
    int devID = findCudaDevice(argc, (const char **)argv);

    // parsing the number of random numbers to generate
    int rand_n = DEFAULT_RAND_N;
    if( checkCmdLineFlag(argc, (const char**) argv, "count") )  
    {       
        rand_n = getCmdLineArgumentInt(argc, (const char**) argv, "count"); 
    }
    printf("Allocating data for %i samples...\n", rand_n);
     
    // parsing the seed
    int seed = DEFAULT_SEED;
    if( checkCmdLineFlag(argc, (const char**) argv, "seed") ) 
    {       
        seed = getCmdLineArgumentInt(argc, (const char**) argv, "seed"); 
    }
    printf("Seeding with %i ...\n", seed);
    

    float *d_Rand; 
    checkCudaErrors( cudaMalloc((void **)&d_Rand, rand_n * sizeof(float)) );
    
    curandGenerator_t prngGPU;
    checkCurandErrors( curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32) ); 
    checkCurandErrors( curandSetPseudoRandomGeneratorSeed(prngGPU, seed) );

    curandGenerator_t prngCPU;
    checkCurandErrors( curandCreateGeneratorHost(&prngCPU, CURAND_RNG_PSEUDO_MTGP32) ); 
    checkCurandErrors( curandSetPseudoRandomGeneratorSeed(prngCPU, seed) );

    //
    // Example 1: Compare random numbers generated on GPU and CPU
    float *h_RandGPU  = (float *)malloc(rand_n * sizeof(float));

    printf("Generating random numbers on GPU...\n\n");
    checkCurandErrors( curandGenerateUniform(prngGPU, (float*) d_Rand, rand_n) );

    printf("\nReading back the results...\n");
    checkCudaErrors( cudaMemcpy(h_RandGPU, d_Rand, rand_n * sizeof(float), cudaMemcpyDeviceToHost) );

    
    float *h_RandCPU  = (float *)malloc(rand_n * sizeof(float));
     
    printf("Generating random numbers on CPU...\n\n");
    checkCurandErrors( curandGenerateUniform(prngCPU, (float*) h_RandCPU, rand_n) ); 
 
    printf("Comparing CPU/GPU random numbers...\n\n");
    float L1norm = compareResults(rand_n, h_RandGPU, h_RandCPU); 
    
    //
    // Example 2: Timing of random number generation on GPU
    const int numIterations = 10;
    int i;
    StopWatchInterface *hTimer;

    checkCudaErrors( cudaDeviceSynchronize() );
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (i = 0; i < numIterations; i++)
    {
        checkCurandErrors( curandGenerateUniform(prngGPU, (float*) d_Rand, rand_n) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer);

    double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer)/(double)numIterations;

    printf("MersenneTwister, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers\n", 
               1.0e-9 * rand_n / gpuTime, gpuTime, rand_n); 

    printf("Shutting down...\n");

    checkCurandErrors( curandDestroyGenerator(prngGPU) );
    checkCurandErrors( curandDestroyGenerator(prngCPU) );
    checkCudaErrors( cudaFree(d_Rand) );
    sdkDeleteTimer( &hTimer);
    free(h_RandGPU);
    free(h_RandCPU);

    cudaDeviceReset();	
    shrQAFinishExit(argc, (const char**)argv, (L1norm < 1e-6) ? QA_PASSED : QA_FAILED);
}


float compareResults(int rand_n, float* h_RandGPU, float* h_RandCPU)
{
    int i;
    float rCPU, rGPU, delta;
    float max_delta = 0.;
    float sum_delta = 0.;
    float sum_ref   = 0.;
    for(i = 0; i < rand_n; i++)
    {
        rCPU = h_RandCPU[i];
        rGPU = h_RandGPU[i];
        delta = fabs(rCPU - rGPU);
        sum_delta += delta;
        sum_ref   += fabs(rCPU);
        if(delta >= max_delta) max_delta = delta;
    }
    float L1norm = (float)(sum_delta / sum_ref);
    printf("Max absolute error: %E\n", max_delta);
    printf("L1 norm: %E\n\n", L1norm);

    return L1norm;
}
