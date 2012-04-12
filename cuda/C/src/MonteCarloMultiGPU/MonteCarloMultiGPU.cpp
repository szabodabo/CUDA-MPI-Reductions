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
 * This sample evaluates fair call price for a
 * given set of European options using Monte Carlo approach.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// includes, project
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrQATest.h>
#include <multithreading.h>

#include "MonteCarlo_common.h"

int *pArgc = NULL;
char **pArgv = NULL;

#ifdef WIN32
#define strcasecmp strcmpi
#endif

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


////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
float randFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////
extern "C" void MonteCarloCPU(
    TOptionValue&   callValue,
    TOptionData optionData,
    float *h_Random,
    int pathN
);

//Black-Scholes formula for call options
extern "C" void BlackScholesCall(
    float& CallResult,
    TOptionData optionData
);


////////////////////////////////////////////////////////////////////////////////
// GPU-driving host thread
////////////////////////////////////////////////////////////////////////////////
//Timer
const int MAX_GPU_COUNT = 8;
StopWatchInterface *hTimer[MAX_GPU_COUNT];

static CUT_THREADPROC solverThread(TOptionPlan *plan)
{
    //Init GPU
    checkCudaErrors( cudaSetDevice(plan->device) );

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, plan->device));

    //Start the timer
    sdkStartTimer(&hTimer[plan->device]);

    // Allocate intermediate memory for MC integrator and initialize
    // RNG states
    initMonteCarloGPU(plan);

    // Main commputation
    MonteCarloGPU(plan);
 
    checkCudaErrors( cudaDeviceSynchronize() );

    //Stop the timer
    sdkStopTimer(&hTimer[plan->device]);

    //Shut down this GPU
    closeMonteCarloGPU(plan);

    cudaStreamSynchronize(0);

    printf("solverThread() finished - GPU Device %d: %s\n", plan->device, deviceProp.name );
    cudaDeviceReset();
    CUT_THREADEND;
}

static void multiSolver(TOptionPlan *plan, int nPlans ){

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*) malloc(nPlans * sizeof(cudaStream_t));
    cudaEvent_t *events = (cudaEvent_t*)malloc(nPlans * sizeof(cudaEvent_t));
    for(int i = 0; i < nPlans; i++) {
        checkCudaErrors( cudaSetDevice(plan[i].device) );
        checkCudaErrors( cudaStreamCreate(&(streams[i])) );
        checkCudaErrors( cudaEventCreate(&(events[i])) );
    }

    //Init Each GPU
    // In CUDA 4.0 we can call cudaSetDevice multiple times to target each device
    // Set the device desired, then perform initializations on that device
   
    for( int i=0 ; i<nPlans ; i++ )  {
        // set the target device to perform initialization on
        checkCudaErrors( cudaSetDevice(plan[i].device) );
    
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, plan[i].device));

        // Allocate intermediate memory for MC integrator
        // and initialize RNG state
        initMonteCarloGPU(&plan[i]);
    }
    //Start the timer
    sdkResetTimer(&hTimer[0]);
    sdkStartTimer(&hTimer[0]);

    for( int i=0; i<nPlans; i++ ) {
        checkCudaErrors( cudaSetDevice(plan[i].device) );

        //Main computations
        MonteCarloGPU(&plan[i], streams[i]);

        checkCudaErrors( cudaEventRecord( events[i] ) );
    }

    for( int i=0; i<nPlans; i++ ) {
        checkCudaErrors( cudaSetDevice(plan[i].device) );
        cudaEventSynchronize( events[i] );
    }
    //Stop the timer
    sdkStopTimer(&hTimer[0]);

    for( int i=0 ; i<nPlans ; i++ ) {
        checkCudaErrors( cudaSetDevice(plan[i].device) );
        closeMonteCarloGPU(&plan[i]);
    }
}



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
#define DO_CPU
#undef DO_CPU

#define PRINT_RESULTS
#undef PRINT_RESULTS


void usage()
{
    printf("--method=[threaded,streamed] [--help]\n");
    printf("Method=threaded: 1 CPU thread for each GPU     (applies for CUDA 3.0 and older)\n");
    printf("       streamed: 1 CPU thread handles all GPUs (requires CUDA 4.0 or newer)\n");
}


int main(int argc, char **argv)
{
    char *multiMethodChoice;
    bool use_threads = true;
    bool bqatest = false;

	pArgc = &argc;
	pArgv = argv;

    shrQAStart(argc, argv);

    if (checkCmdLineFlag(argc, (const char **)argv, "qatest") ) {
        bqatest = true;
    }

    getCmdLineArgumentString(argc, (const char **)argv, "method", &multiMethodChoice);
    if( checkCmdLineFlag( argc, (const char **)argv, "h") || 
        checkCmdLineFlag(argc, (const char **)argv, "help") ) 
    {
        usage();
        exit(0);
    }

    if( multiMethodChoice == NULL ) {
        use_threads = true;
    } else {
        if(!strcasecmp(multiMethodChoice, "threaded"))
            use_threads = true;
        else
            use_threads = false;
    }
    if( use_threads == false ) { 
        printf("Using single CPU thread for multiple GPUs\n");
    }
   

    // determine runtime version
    int runtimeVersion;
    cudaRuntimeGetVersion( &runtimeVersion );

    const int         OPT_N = 256;
    const int        PATH_N = 1 << 18;
    const unsigned long long SEED = 777;

    //Input data array
    TOptionData optionData[OPT_N];
    //Final GPU MC results
    TOptionValue callValueGPU[OPT_N];
    //"Theoretical" call values by Black-Scholes formula
    float callValueBS[OPT_N];
    //Solver config
    TOptionPlan optionSolver[MAX_GPU_COUNT];
    //OS thread ID
    CUTThread threadID[MAX_GPU_COUNT];

    //GPU number present in the system
    int GPU_N;
    int gpuBase, gpuIndex;
    int i;

    float time;

    double delta, ref, sumDelta, sumRef, sumReserve;

    checkCudaErrors( cudaGetDeviceCount(&GPU_N) );
    for( int i=0; i<GPU_N; i++ ) {
       sdkCreateTimer(&hTimer[i]);
       sdkResetTimer(&hTimer[i]);
    }

#ifdef _EMU
	GPU_N = 1;
#endif
    printf("main(): generating input data...\n");
        srand(123);
        for(i=0; i < OPT_N; i++){
            optionData[i].S = randFloat(5.0f, 50.0f);
            optionData[i].X = randFloat(10.0f, 25.0f);
            optionData[i].T = randFloat(1.0f, 5.0f);
            optionData[i].R = 0.06f;
            optionData[i].V = 0.10f;
            callValueGPU[i].Expected   = -1.0f;
            callValueGPU[i].Confidence = -1.0f;
        }

    printf("main(): starting %i host threads...\n", GPU_N);
        //Get option count for each GPU
        for(i = 0; i < GPU_N; i++)
            optionSolver[i].optionCount = OPT_N / GPU_N;
        //Take into account cases with "odd" option counts
        for(i = 0; i < (OPT_N % GPU_N); i++)
            optionSolver[i].optionCount++;

        //Assign GPU option ranges
        gpuBase = 0;
        for(i = 0; i < GPU_N; i++){
            optionSolver[i].device     = i;
            optionSolver[i].optionData = optionData   + gpuBase;
            optionSolver[i].callValue  = callValueGPU + gpuBase;
            // all devices you the same global seed, but start
            // the sequence at a different offset
            optionSolver[i].seed       = SEED;
            optionSolver[i].pathN      = PATH_N;
            gpuBase += optionSolver[i].optionCount;
        }


    if( use_threads || bqatest ) {
        //Start CPU thread for each GPU
        for(gpuIndex = 0; gpuIndex < GPU_N; gpuIndex++)
            threadID[gpuIndex] = cutStartThread((CUT_THREADROUTINE)solverThread, &optionSolver[gpuIndex]);
        printf("main(): waiting for GPU results...\n");
        cutWaitForThreads(threadID, GPU_N);
        //Stop the timer


    printf("main(): GPU statistics, threaded\n");
        for(i = 0; i < GPU_N; i++){
            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, optionSolver[i].device));
            printf("GPU Device #%i: %s\n", optionSolver[i].device, deviceProp.name);
            printf("Options         : %i\n", optionSolver[i].optionCount);
            printf("Simulation paths: %i\n", optionSolver[i].pathN);
            time = sdkGetTimerValue(&hTimer[i]);
            printf("Total time (ms.): %f\n", time);
            printf("Options per sec.: %f\n", OPT_N / (time * 0.001));
        }

    printf("main(): comparing Monte Carlo and Black-Scholes results...\n");
        sumDelta   = 0;
        sumRef     = 0;
        sumReserve = 0;
        for(i = 0; i < OPT_N; i++){
            BlackScholesCall( callValueBS[i], optionData[i] );
            delta     = fabs(callValueBS[i] - callValueGPU[i].Expected);
            ref       = callValueBS[i];
            sumDelta += delta;
            sumRef   += fabs(ref);
            if(delta > 1e-6) sumReserve += callValueGPU[i].Confidence / delta;
#ifdef PRINT_RESULTS
            printf("BS: %f; delta: %E\n", callValueBS[i], delta);
#endif

        }
        sumReserve /= OPT_N;
    }

    if( !use_threads || bqatest )  
    {
        multiSolver( optionSolver, GPU_N );

        printf("main(): GPU statistics, streamed\n");
        for(i = 0; i < GPU_N; i++){
            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, optionSolver[i].device));
            printf("GPU Device #%i: %s\n", optionSolver[i].device, deviceProp.name);
            printf("Options         : %i\n", optionSolver[i].optionCount);
            printf("Simulation paths: %i\n", optionSolver[i].pathN);
        }
        time = sdkGetTimerValue(&hTimer[0]);
        printf("\nTotal time (ms.): %f\n", time);
        printf("\tNote: This is elapsed time for all to compute.\n");
        printf("Options per sec.: %f\n", OPT_N / (time * 0.001));

        printf("main(): comparing Monte Carlo and Black-Scholes results...\n");
        sumDelta   = 0;
        sumRef     = 0;
        sumReserve = 0;
        for(i = 0; i < OPT_N; i++){
            BlackScholesCall( callValueBS[i], optionData[i] );
            delta     = fabs(callValueBS[i] - callValueGPU[i].Expected);
            ref       = callValueBS[i];
            sumDelta += delta;
            sumRef   += fabs(ref);
            if(delta > 1e-6) sumReserve += callValueGPU[i].Confidence / delta;
#ifdef PRINT_RESULTS
            printf("BS: %f; delta: %E\n", callValueBS[i], delta);
#endif
        }
        sumReserve /= OPT_N;
    }

#ifdef DO_CPU
    printf("main(): running CPU MonteCarlo...\n");
        TOptionValue callValueCPU;
        sumDelta = 0;
        sumRef   = 0;
        for(i = 0; i < OPT_N; i++){
            MonteCarloCPU(
                callValueCPU,
                optionData[i],
                NULL,
                PATH_N
            );
            delta     = fabs(callValueCPU.Expected - callValueGPU[i].Expected);
            ref       = callValueCPU.Expected;
            sumDelta += delta;
            sumRef   += fabs(ref);
            printf("Exp : %f | %f\t", callValueCPU.Expected,   callValueGPU[i].Expected);
            printf("Conf: %f | %f\n", callValueCPU.Confidence, callValueGPU[i].Confidence);
        }
    printf("L1 norm: %E\n", sumDelta / sumRef);
#endif

    printf("Shutting down...\n");
	for( int i=0; i<GPU_N; i++ ) {	
		sdkDeleteTimer(&hTimer[i]);
        checkCudaErrors( cudaSetDevice(i) );
        cudaDeviceReset();
    }

    printf("Test Summary...\n");
    printf("L1 norm        : %E\n", sumDelta / sumRef);
    printf("Average reserve: %f\n", sumReserve);
    shrQAFinishExit(argc, (const char **)argv, (sumReserve > 1.0f) ? QA_PASSED : QA_FAILED);
}
