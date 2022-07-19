#include "reduction.cuh"

// ============ minElement ====================
__inline__ __device__ void warpReduceMin(volatile int *pVal, volatile int *pIndex)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        int shflVal = __shfl_down_sync(warpSize - 1, *pVal, offset);
        int shfIndex = __shfl_down_sync(warpSize - 1, *pIndex, offset);
        if (shflVal < *pVal)
        {
            *pVal = shflVal;
            *pIndex = shfIndex;
        }
    }
}

__inline__ __device__ void blockReduceMin(volatile int *pVal, volatile int *pIndex)
{
    static __shared__ int sdata[32];
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
__global__ void deviceReduceKernel(int* g_values, int* g_index, int N)
{
    int minVal = INT_MAX;
    int minIndex = -1;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x
    )
    {
        int candidate = g_values[i];
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

TYPE minElement(TYPE* g_vet, unsigned int size, unsigned int* outIndex)
{
    return (TYPE)0.0;
}

// ============ reduction with atomic ====================
template <bool minimum>
__inline__ __device__ void warpReduce(volatile int *pVal)
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
__inline__ __device__ void blockReduce(volatile int *pVal)
{
    static __shared__ int sdata[32];

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
__global__ void deviceReduceBlockAtomicKernel(int* g_data, int N)
{
    int partial = minimum ? INT_MAX : INT_MIN;

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
    return true;
}