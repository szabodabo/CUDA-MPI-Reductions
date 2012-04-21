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
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <stdio.h>

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
sumreduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            mySum += g_idata[i+blockSize];  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
minreduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            mySum += g_idata[i+blockSize];  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}


template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
maxreduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            mySum += g_idata[i+blockSize];  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}


extern "C"
bool isPow2(unsigned int x);


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
sumreduce(int size, int threads, int blocks, 
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
    case 0:
        break;
    case 1:
        break;
    case 2:
        break;
    case 3:
        break;
    case 4:
        break; 
    case 5:
        break;       
    case 6:
    default:
        if (isPow2(size))
        {
            switch (threads)
            {
            case 512:
                sumreduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 256:
                sumreduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 128:
                sumreduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 64:
                sumreduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 32:
                sumreduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 16:
                sumreduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  8:
                sumreduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  4:
                sumreduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  2:
                sumreduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  1:
                sumreduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
        }
        else
        {
            switch (threads)
            {
            case 512:
                sumreduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 256:
                sumreduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 128:
                sumreduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 64:
                sumreduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 32:
                sumreduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 16:
                sumreduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  8:
                sumreduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  4:
                sumreduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  2:
                sumreduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  1:
                sumreduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
        }
        break;       
    }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
minreduce(int size, int threads, int blocks, 
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
    case 0:
        break;
    case 1:
        break;
    case 2:
        break;
    case 3:
        break;
    case 4:
        break; 
    case 5:
        break;       
    case 6:
    default:
        if (isPow2(size))
        {
					//min
            switch (threads)
            {
            case 512:
                minreduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 256:
                minreduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 128:
                minreduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 64:
                minreduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 32:
                minreduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 16:
                minreduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  8:
                minreduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  4:
                minreduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  2:
                minreduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  1:
                minreduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
        }
        else
        {
            switch (threads)
            {
            case 512:
                minreduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 256:
                minreduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 128:
                minreduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 64:
                minreduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 32:
                minreduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 16:
                minreduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  8:
                minreduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  4:
                minreduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  2:
                minreduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  1:
                minreduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
        }
        break;       
    }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
maxreduce(int size, int threads, int blocks, 
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
    case 0:
        break;
    case 1:
        break;
    case 2:
        break;
    case 3:
        break;
    case 4:
        break; 
    case 5:
        break;       
    case 6:
    default:
        if (isPow2(size))
        {
					//max
            switch (threads)
            {
            case 512:
                maxreduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 256:
                maxreduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 128:
                maxreduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 64:
                maxreduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 32:
                maxreduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 16:
                maxreduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  8:
                maxreduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  4:
                maxreduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  2:
                maxreduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  1:
                maxreduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
        }
        else
        {
            switch (threads)
            {
            case 512:
                maxreduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 256:
                maxreduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 128:
                maxreduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 64:
                maxreduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 32:
                maxreduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 16:
                maxreduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  8:
                maxreduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  4:
                maxreduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  2:
                maxreduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  1:
                maxreduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
        }
        break;       
    }
}


// Instantiate the reduction function for 3 types
template void 
sumreduce<int>(int size, int threads, int blocks, 
            int whichKernel, int *d_idata, int *d_odata);

template void 
sumreduce<float>(int size, int threads, int blocks, 
              int whichKernel, float *d_idata, float *d_odata);
              
template void 
sumreduce<double>(int size, int threads, int blocks, 
               int whichKernel, double *d_idata, double *d_odata);

// Instantiate the reduction function for 3 types
template void 
minreduce<int>(int size, int threads, int blocks, 
            int whichKernel, int *d_idata, int *d_odata);

template void 
minreduce<float>(int size, int threads, int blocks, 
              int whichKernel, float *d_idata, float *d_odata);
              
template void 
minreduce<double>(int size, int threads, int blocks, 
               int whichKernel, double *d_idata, double *d_odata);

// Instantiate the reduction function for 3 types
template void 
maxreduce<int>(int size, int threads, int blocks, 
            int whichKernel, int *d_idata, int *d_odata);

template void 
maxreduce<float>(int size, int threads, int blocks, 
              int whichKernel, float *d_idata, float *d_odata);
              
template void 
maxreduce<double>(int size, int threads, int blocks, 
               int whichKernel, double *d_idata, double *d_odata);



#endif // #ifndef _REDUCE_KERNEL_H_
