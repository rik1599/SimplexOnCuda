#include "generator.cuh"
#include <curand.h>
#include <curand_kernel.h>

/*==================== Kernel per la generazione =====================*/

// per avere ripetibilità è necessario che stesso seed generi la stessa matrice
// non possiamo dipendere dalla dimensione del blocco e dal numero di blocchi per grid per generare i numeri
// utilizzando questa soluzione abbiamo 2^64 matrici diverse che vengono generate
// idealmente si vorrebbe utilizzare un seed solo ed utilizzare l'offset della sequenza in base alla posizione,
// facendo così però serve troppo tempo per la generazione
// visto il fine del generatore ha senso massimizzare la velocità sommando al seed in input la posizione globale ed utilizzare il risultato come seed della generazione
__global__ void generateMatrix(TYPE *outMatrix, int rows, int cols, size_t pitch, unsigned long long seed, double minimum, double maximum)
{
    curandState state;
    for (int idY = blockDim.y * blockIdx.y + threadIdx.y; idY < rows; idY += blockDim.y * gridDim.y)
    { // Grid stride loop per migliore scalabilità
        for (int idX = blockDim.x * blockIdx.x + threadIdx.x; idX < cols; idX += blockDim.x * gridDim.x)
        {
            unsigned long long globalPosition = cols * idY + idX;
            curand_init(seed + globalPosition, 0, 0, &state);
            *(INDEX(outMatrix, idY, idX, pitch)) = (curand_uniform(&state) * (maximum - minimum)) + minimum;
        }
    }
}

// discorso equivalente a quello sopra
__global__ void generateVector(TYPE *outVet, int size, unsigned long long seed, double minimum, double maximum)
{
    for (int id = blockDim.x * blockIdx.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x)
    {
        curandState state;
        curand_init(seed + id, 0, 0, &state);
        outVet[id] = (curand_uniform(&state) * (maximum - minimum)) + minimum;
    }
}

/*================== Funzioni per la generazione =======================*/

void generateVectorInParallel(TYPE *dst, int size, unsigned long long seed, double minimum, double maximum)
{
    TYPE *dev_array;

    HANDLE_ERROR(cudaMalloc(
        (void **)&dev_array,
        BYTE_SIZE(size)));

    runVectorGenerationKernel(dev_array, size, seed, NULL, minimum, maximum);

    HANDLE_ERROR(cudaMemcpy(dst, dev_array, BYTE_SIZE(size), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_array));
}

// generazione vettore in parallelo asincrono
cudaStream_t *generateVectorInParallelAsync(TYPE *dst, int size, unsigned long long seed, double minimum, double maximum)
{
    TYPE *dev_array;

    // stream per il parallelismo dei task
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
    HANDLE_ERROR(cudaStreamCreate(stream));

    HANDLE_ERROR(cudaMallocAsync(
        (void **)&dev_array,
        BYTE_SIZE(size),
        *stream));

    runVectorGenerationKernel(dev_array, size, seed, stream, minimum, maximum);

    HANDLE_ERROR(cudaMemcpyAsync(dst, dev_array, BYTE_SIZE(size), cudaMemcpyDeviceToHost, *stream));
    HANDLE_ERROR(cudaFreeAsync(dev_array, *stream));
    return stream;
}

// linearizza per righe, per avere una linearizzata per colonne generare la trasposta
void generateMatrixInParallel(TYPE *dst, int width, int height, unsigned long long seed, double minimum, double maximum)
{
    size_t pitch;
    TYPE *dev_matrix;

    // allocazione in memoria device, prestare attenzione al fatto che è linearizzata per colonne
    HANDLE_ERROR(cudaMallocPitch(
        (void **)&dev_matrix, // puntatore device
        &pitch,               // puntatore pitch
        BYTE_SIZE(width),     // dimensione in bite della larghezza della matrice, bytesize dei constraints dato che linearizzata per colonne
        height                // l'altezza, dato che linearizzata per colonne è il numero di variabili
        ));

    // Lancio del kernel
    runMatrixGenerationKernel(dev_matrix, width, height, pitch, seed, NULL, minimum, maximum);

    HANDLE_ERROR(cudaMemcpy2D(
        dst,
        BYTE_SIZE(width),
        dev_matrix,
        pitch,
        BYTE_SIZE(width),
        height,
        cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_matrix));
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

    runMatrixGenerationKernel(dev_matrix, width, height, pitch, seed, stream, minimum, maximum);

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

void runMatrixGenerationKernel(TYPE *dev_ptr, int width, int height, size_t pitch, unsigned long long seed, cudaStream_t *stream, double minimum, double maximum)
{
    dim3 block2D = dim3(TILE_X * TILE_Y > 1024 ? 32 : TILE_X,
                        TILE_X * TILE_Y > 1024 ? 32 : TILE_Y); // limita il max numero di kernel per block a 1024

    dim3 grid2D = dim3(((block2D.x - 1) + height) / block2D.x, ((block2D.y - 1) + width) / block2D.y);
    grid2D = dim3(grid2D.x <= GRID_2D_MAX_X ? grid2D.x : GRID_2D_MAX_X,
                  grid2D.y <= GRID_2D_MAX_Y ? grid2D.y : GRID_2D_MAX_Y);

    generateMatrix<<<grid2D, block2D, 0, stream ? *stream : 0>>>(dev_ptr, height, width, pitch, seed, minimum, maximum); // se non si passa uno stream utilizza quello 0, quindi quello default
    HANDLE_KERNEL_ERROR();
}

void runVectorGenerationKernel(TYPE *dev_ptr, int size, unsigned long long seed, cudaStream_t *stream, double minimum, double maximum)
{
    int blockSize = LINEAR_BLOCK > 1024 ? 1024 : LINEAR_BLOCK;
    int gridSize = (size + blockSize - 1) / blockSize;
    gridSize = gridSize < LINEAR_GRID_MAX ? gridSize : LINEAR_GRID_MAX; // il grid stride si occupa del secondo caso
    generateVector<<<gridSize, blockSize, 0, stream ? *stream : 0>>>(dev_ptr, size, seed, minimum, maximum);
    HANDLE_KERNEL_ERROR();
}