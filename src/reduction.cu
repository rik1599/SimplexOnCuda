#include "reduction.cuh"
#include "error.cuh"

#define THREADS 512
#define BL(N) min((N + THREADS - 1) / THREADS, 1024)

// ============ minElement ====================
__inline__ __device__ void warpReduceMin(volatile TYPE *pVal, volatile int *pIndex)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        TYPE shflVal = __shfl_down_sync(warpSize - 1, *pVal, offset);
        int shfIndex = __shfl_down_sync(warpSize - 1, *pIndex, offset);
        if (shflVal < *pVal)
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

    *pVal = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : INT_MAX;
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
        if (candidate < minVal)
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

template <bool isFirstExecution>
__global__ void deviceReduceKernel(TYPE* knownTerms, TYPE* rowPivot, unsigned int* g_index, int N)
{
    TYPE minVal = INT_MAX * 1.0;
    int minIndex = -1;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x
    )
    {
        TYPE candidate = knownTerms[i]/rowPivot[i];
        if (candidate < minVal)
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
        knownTerms[blockIdx.x] = minVal;
        g_index[blockIdx.x] = minIndex;
    }
}

TYPE minElement(TYPE* g_vet, unsigned int size, unsigned int* outIndex)
{
    unsigned int* g_index;
    HANDLE_ERROR(cudaMalloc((void**)&g_index, BL(size) * sizeof(unsigned int)));

    deviceReduceKernel<true><<<BL(size), THREADS>>>(g_vet, g_index, size);
    if (BL(size) > 1)
    {
        deviceReduceKernel<false><<<1, 1024>>>(g_vet, g_index, BL(size));
    }
    HANDLE_KERNEL_ERROR();

    TYPE parallelMin;
    HANDLE_ERROR(cudaMemcpy(&parallelMin, g_vet, sizeof(int), cudaMemcpyDefault));
    HANDLE_ERROR(cudaMemcpy(outIndex, g_index, sizeof(int), cudaMemcpyDefault));

    cudaFree(g_index);

    return parallelMin;
}

TYPE minElement(TYPE* knownTerms, TYPE* rowPivot, unsigned int size, unsigned int* outIndex)
{
    unsigned int* g_index;
    HANDLE_ERROR(cudaMalloc((void**)&g_index, BL(size) * sizeof(unsigned int)));

    deviceReduceKernel<true><<<BL(size), THREADS>>>(knownTerms, rowPivot, g_index, BL(size));
    if (BL(size) > 1)
    {
        deviceReduceKernel<false><<<1, 1024>>>(knownTerms, rowPivot, g_index, BL(size));
    }
    HANDLE_KERNEL_ERROR();

    TYPE parallelMin;
    HANDLE_ERROR(cudaMemcpy(&parallelMin, knownTerms, sizeof(int), cudaMemcpyDefault));
    HANDLE_ERROR(cudaMemcpy(outIndex, g_index, sizeof(int), cudaMemcpyDefault));

    cudaFree(g_index);

    return parallelMin;
}

// ============ reduction with atomic ====================
template <bool minimum>
__inline__ __device__ void warpReduce(volatile TYPE *pVal)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {   
        if (minimum)
            *pVal = min(*pVal, __shfl_down_sync(warpSize - 1, *pVal, offset));
        else
            *pVal = max(*pVal, __shfl_down_sync(warpSize - 1, *pVal, offset));
    }
}

template <bool minimum>
__inline__ __device__ void blockReduce(volatile TYPE *pVal)
{
    static __shared__ TYPE sdata[32];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduce<minimum>(pVal);

    if (lane == 0)
    {
        sdata[wid] = *pVal;
    }

    __syncthreads();

    if (minimum)
        *pVal = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : INT_MAX;
    else
        *pVal = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : INT_MIN;

    if (wid == 0)
    {
        warpReduce<minimum>(pVal);
    }
}

template <bool minimum>
__global__ void deviceReduceBlockAtomicKernel(TYPE* g_data, unsigned int N)
{
    TYPE partial = minimum ? INT_MAX : INT_MIN;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x
    )
    {
        if (minimum)
            partial = min(partial, g_data[i]);
        else
            partial = max(partial, g_data[i]);
    }

    blockReduce<minimum>(&partial);

    if (threadIdx.x == 0)
    {
        if (minimum)
            atomicMin(g_data, partial);
        else
            atomicMax(g_data, partial);
    }
}

bool isGreaterThanZero(TYPE* g_vet, unsigned int size)
{
    return true;
}

bool isLessThanZero(TYPE* g_vet, unsigned int size)
{
    TYPE* g_vetCpy;
    HANDLE_ERROR(cudaMalloc((void**)&g_vetCpy, BYTE_SIZE(size)));
    HANDLE_ERROR(cudaMemcpy(g_vetCpy, g_vet, size, cudaMemcpyDefault));

    deviceReduceBlockAtomicKernel<false><<<BL(size), THREADS>>>(g_vetCpy, size);
    if (BL(size) > 1)
    {
        deviceReduceBlockAtomicKernel<false><<<1, 1024>>>(g_vetCpy, size);
    }
    HANDLE_KERNEL_ERROR();

    TYPE maximum = *g_vetCpy;
    HANDLE_ERROR(cudaFree(g_vetCpy));
    
    return (maximum <= 0);
}