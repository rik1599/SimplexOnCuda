#include "generator.cuh"
#include <curand.h>
#include <curand_kernel.h>

#define THREADS 512
#define BL(N) min((N + THREADS - 1) / THREADS, 1024)

/*==================== Kernel per la generazione =====================*/
__global__ void generateMatrixLinear(TYPE *outMatrix, int rows, int cols, size_t pitch, unsigned long long seed, double minimum, double maximum)
{
    curandState state;
        // Grid stride loop per migliore scalabilità
    for (int idX = blockDim.x * blockIdx.x + threadIdx.x; idX < cols; idX += blockDim.x * gridDim.x)
    {
        curand_init(seed, 0, idX * rows, &state);
        //ogni kernel lavora nella sua colonna
        for(int i = 0; i < rows; i++){
                *(INDEX(outMatrix, i, idX, pitch)) = (curand_uniform(&state) * (maximum - minimum)) + minimum;
        }
    }
}

// discorso equivalente a quello sopra
__global__ void generateVector(TYPE *outVet, int size, unsigned long long seed, double minimum, double maximum)
{
    curandState state;
    for (int id = blockDim.x * blockIdx.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x)
    {
        curand_init(seed, 0, id, &state);
        outVet[id] = (curand_uniform(&state) * (maximum - minimum)) + minimum;
    }
}

/*================== Funzioni per la generazione =======================*/

cudaStream_t *generateVectorInParallelAsync(TYPE *dev_array, int size, unsigned long long seed, double minimum, double maximum)
{
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
    HANDLE_ERROR(cudaStreamCreate(stream));

    generateVector<<<BL(size), THREADS, 0, *stream>>>(dev_array, size, seed, minimum, maximum);
    
    return stream;
}

cudaStream_t *generateMatrixInParallelAsync(TYPE *dst, int width, int height, unsigned long long seed, double minimum, double maximum)
{
    size_t pitch;
    TYPE *dev_matrix;

    // stream per il parallelismo dei task
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
    HANDLE_ERROR(cudaStreamCreate(stream));

    // allocazione in memoria device
    HANDLE_ERROR(cudaMallocPitch(
        (void **)&dev_matrix, // puntatore device
        &pitch,               // puntatore pitch
        BYTE_SIZE(width),     // dimensione in bite della larghezza della matrice, bytesize dei constraints dato che linearizzata per colonne
        height                // l'altezza, dato che linearizzata per colonne è il numero di variabili
        ));

    generateMatrixLinear<<<BL(width), THREADS, 0, *stream>>>(dev_matrix, height, width, pitch, seed, minimum, maximum); // se non si passa uno stream utilizza quello 0, quindi quello default

    HANDLE_ERROR(cudaMemcpy2DAsync(
        dst,
        BYTE_SIZE(width),
        dev_matrix,
        pitch,
        BYTE_SIZE(width),
        height,
        cudaMemcpyDeviceToHost,
        *stream));

    HANDLE_ERROR(cudaFreeAsync(dev_matrix, *stream));
    return stream;
}