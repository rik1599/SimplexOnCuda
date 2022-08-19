#include "reduction.cuh"
#include "error.cuh"
#include "stdio.h"

#define THREADS 512
#define BL(N) min((N + THREADS - 1) / THREADS, 1024)

// ============ minElement ====================
__inline__ __device__ void warpReduceMin(volatile TYPE *pVal, volatile int *pIndex)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        TYPE shflVal = __shfl_down_sync(warpSize - 1, *pVal, offset);
        int shfIndex = __shfl_down_sync(warpSize - 1, *pIndex, offset);
        if (compare(shflVal, *pVal) < 0)
        {
            *pVal = shflVal;
            *pIndex = shfIndex;
        }
    }
}

__inline__ __device__ void blockReduceMin(volatile TYPE *pVal, volatile int *pIndex)
{
    static __shared__ TYPE sdata[32];
    static __shared__ int sindex[32];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduceMin(pVal, pIndex);

    if (lane == 0)
    {
        sdata[wid] = *pVal;
        sindex[wid] = *pIndex;
    }

    __syncthreads();

    *pVal = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : INT_MAX * 1.0;
    *pIndex = (threadIdx.x < blockDim.x / warpSize) ? sindex[lane] : -1;

    if (wid == 0)
    {
        warpReduceMin(pVal, pIndex);
    }
}

template <bool isFirstExecution>
__global__ void deviceReduceKernel(TYPE* g_values, unsigned int* g_index, int N)
{
    TYPE minVal = INT_MAX * 1.0;
    int minIndex = -1;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x
    )
    {
        TYPE candidate = g_values[i];
        if (compare(candidate, minVal) < 0)
        {
            minVal = candidate;

            if (isFirstExecution)
                minIndex = i;
            else
                minIndex = g_index[i];
        }
    }

    blockReduceMin(&minVal, &minIndex);

    if (threadIdx.x == 0)
    {
        g_values[blockIdx.x] = minVal;
        g_index[blockIdx.x] = minIndex;
    }
}

TYPE minElement(TYPE* g_vet, unsigned int size, unsigned int* outIndex)
{
    unsigned int* g_index;
    HANDLE_ERROR(cudaMalloc((void**)&g_index, BL(size) * sizeof(unsigned int)));

    TYPE* g_vetCpy;
    HANDLE_ERROR(cudaMalloc((void**)&g_vetCpy, BYTE_SIZE(size)));
    HANDLE_ERROR(cudaMemcpy(g_vetCpy, g_vet, BYTE_SIZE(size), cudaMemcpyDefault));

    deviceReduceKernel<true><<<BL(size), THREADS>>>(g_vetCpy, g_index, size);
    if (BL(size) > 1)
    {
        deviceReduceKernel<false><<<1, 1024>>>(g_vetCpy, g_index, BL(size));
    }
    HANDLE_KERNEL_ERROR();

    TYPE parallelMin;
    HANDLE_ERROR(cudaMemcpy(&parallelMin, g_vetCpy, BYTE_SIZE(1), cudaMemcpyDefault));
    HANDLE_ERROR(cudaMemcpy(outIndex, g_index, sizeof(int), cudaMemcpyDefault));

    HANDLE_ERROR(cudaFree(g_index));
    HANDLE_ERROR(cudaFree(g_vetCpy));

    return parallelMin;
}

__global__ void createIndicatorsVector(TYPE* knownTerms, TYPE* rowPivot, unsigned int N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x
    )
    {
        knownTerms[i] = knownTerms[i] / rowPivot[i];
    }
}

TYPE minElement(TYPE* knownTerms, TYPE* rowPivot, unsigned int size, unsigned int* outIndex)
{
    unsigned int* g_index;
    HANDLE_ERROR(cudaMalloc((void**)&g_index, BL(size) * sizeof(unsigned int)));

    TYPE* g_vetCpy;
    HANDLE_ERROR(cudaMalloc((void**)&g_vetCpy, BYTE_SIZE(size)));
    HANDLE_ERROR(cudaMemcpy(g_vetCpy, knownTerms, BYTE_SIZE(size), cudaMemcpyDefault));
    createIndicatorsVector<<<BL(size), THREADS>>>(g_vetCpy, rowPivot, size);
    HANDLE_KERNEL_ERROR();

    deviceReduceKernel<true><<<BL(size), THREADS>>>(g_vetCpy, g_index, size);
    if (BL(size) > 1)
    {
        deviceReduceKernel<false><<<1, 1024>>>(g_vetCpy, g_index, BL(size));
    }
    HANDLE_KERNEL_ERROR();

    TYPE parallelMin;
    HANDLE_ERROR(cudaMemcpy(&parallelMin, g_vetCpy, BYTE_SIZE(1), cudaMemcpyDefault));
    HANDLE_ERROR(cudaMemcpy(outIndex, g_index, sizeof(int), cudaMemcpyDefault));

    HANDLE_ERROR(cudaFree(g_vetCpy));
    HANDLE_ERROR(cudaFree(g_index));

    return parallelMin;
}

// ============ max value-only ====================
__inline__ __device__ void warpReduceMax(volatile TYPE *pVal)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        *pVal = fmax(*pVal, __shfl_down_sync(warpSize - 1, *pVal, offset));
    }
}

__inline__ __device__ void blockReduceMax(volatile TYPE *pVal)
{
    static __shared__ TYPE sdata[32];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduceMax(pVal);

    if (lane == 0)
    {
        sdata[wid] = *pVal;
    }

    __syncthreads();

    *pVal = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : INT_MIN * 1.0;

    if (wid == 0)
    {
        warpReduceMax(pVal);
    }
}

__global__ void deviceReduceKernel(TYPE* g_values, int N)
{
    TYPE maxVal = INT_MIN * 1.0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x
    )
    {
        maxVal = fmax(maxVal, g_values[i]);
    }

    blockReduceMax(&maxVal);

    if (threadIdx.x == 0)
    {
        g_values[blockIdx.x] = maxVal;
    }
}

bool isLessThanZero(TYPE* g_vet, unsigned int size)
{
    TYPE* g_vetCpy;
    HANDLE_ERROR(cudaMalloc((void**)&g_vetCpy, BYTE_SIZE(size)));
    HANDLE_ERROR(cudaMemcpy(g_vetCpy, g_vet, BYTE_SIZE(size), cudaMemcpyDefault));

    deviceReduceKernel<<<BL(size), THREADS>>>(g_vetCpy, size);
    if (BL(size) > 1)
    {
        deviceReduceKernel<<<1, 1024>>>(g_vetCpy, BL(size));
    }
    HANDLE_KERNEL_ERROR();

    TYPE parallelMax;
    HANDLE_ERROR(cudaMemcpy(&parallelMax, g_vetCpy, BYTE_SIZE(1), cudaMemcpyDefault));
    HANDLE_ERROR(cudaFree(g_vetCpy));
    return compare(parallelMax) <= 0;
}