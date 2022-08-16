#include "solver.h"
#include "twoPhaseMethod.h"
#include "reduction.cuh"
#include "error.cuh"

struct matrixInfo
{
    TYPE *mat;
    size_t pitch;
    int rows;
    int cols;
};

#define TILE_DIM 32
// (N - TILE_DIM)/(8192 - TILE_DIM) = (BLOCK_DIM(N) - 1)/15
#define BLOCK_DIM(N) ceil((N + 512.0) / 544.0)

#define THREADS 512
#define BL(N) min((N + THREADS - 1) / THREADS, 1024)

/** Copio una colonna della matrice (con accesso strided) in un vettore in memoria globale
 */
__global__ void copyColumn(matrixInfo matInfo, int colToCpy, TYPE *dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    char *pMat = (char *)matInfo.mat;

    for (; i < matInfo.rows; i += step)
    {
        dst[i] = ((TYPE *)(pMat + i * matInfo.pitch))[colToCpy];
    }
}

/** Costruisce il vettore degli indicatori lastCol/colPivot */
__global__ void createIndicatorVector(matrixInfo matInfo, int colPivotIndex)
{
}

/** 9) Aggiornamento tableau per tile. Per ogni A[y][x]
 *      se y == colPivotIndex allora A[y][x] = A[y][x] * recPivot
 *      altrimenti  A[y][x] = - colPivot[y] * rowPivot[x] * recPivot + A[y][x]
 */
__global__ void updateVariables(matrixInfo matInfo, double *colPivot, double *rowPivot, int colPivotIndex, double pivot)
{
    // coordinate
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // dimensioni griglia
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    double *pRow;
    char *pMat = (char *)matInfo.mat;
    for (int col = x; col < matInfo.cols; col += nx)
    {
        for (int row = y; row < matInfo.rows; row += ny)
        {
            pRow = (double *)(pMat + row * matInfo.pitch);

            pRow[col] = col == colPivotIndex ? __drcp_rd(pivot) * pRow[col] : pRow[col] - (rowPivot[col] * __drcp_rd(pivot) * colPivot[row]);
        }
    }
}

__global__ void updateCostsVector(TYPE* costVector, int size, double *colPivot, double costsPivot, double pivot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (; i < size; i += step)
    {
        costVector[i] = costVector[i] - (costsPivot * __drcp_rd(pivot) * colPivot[i]);
    }
}

__inline__ void updateAll(tabular_t *tabular, TYPE *colPivot, int colPivotIndex, TYPE *rowPivot, TYPE costsPivot)
{
    matrixInfo matInfo = {tabular->table, tabular->pitch, tabular->rows, tabular->cols};

    copyColumn<<<BL(tabular->rows), THREADS>>>(matInfo, colPivotIndex, colPivot);
    HANDLE_KERNEL_ERROR();

    TYPE pivot;
    HANDLE_ERROR(cudaMemcpy(&pivot, rowPivot + colPivotIndex, BYTE_SIZE(1), cudaMemcpyDefault));

    cudaStream_t streams[2];
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(BLOCK_DIM(tabular->cols), BLOCK_DIM(tabular->rows));
    cudaStreamCreate(&streams[0]);
    updateVariables<<<grid, block, 0, streams[0]>>>(matInfo, colPivot, rowPivot, colPivotIndex, pivot);

    cudaStreamCreate(&streams[1]);
    updateCostsVector<<<BL(tabular->rows), THREADS, 0, streams[1]>>>(tabular->costsVector, tabular->rows, colPivot, costsPivot, pivot);

    HANDLE_KERNEL_ERROR();

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}

int solve(tabular_t *tabular, int *base)
{
    TYPE *rowPivot, *colPivot;
    HANDLE_ERROR(cudaMalloc((void **)&rowPivot, BYTE_SIZE(tabular->cols)));
    HANDLE_ERROR(cudaMalloc((void **)&colPivot, BYTE_SIZE(tabular->rows)));

    unsigned int colPivotIndex;
    unsigned int rowPivotIndex;
    TYPE costsPivot;

    while ((costsPivot = minElement(tabular->costsVector + 1, tabular->cols - 1, &rowPivotIndex)) < 0)
    {
        //minElement ritorna un indice da 0 a tabular->cols-2
        rowPivotIndex++;

        TYPE *pRowPivot = (double *)((char *)tabular->constraintsMatrix + rowPivotIndex * tabular->pitch);

        HANDLE_ERROR(cudaMemcpy(rowPivot, pRowPivot, BYTE_SIZE(tabular->cols), cudaMemcpyDefault));

        if (isLessThanZero(rowPivot, tabular->cols))
        {
            return UNBOUNDED;
        }

        updateAll(tabular, colPivot, colPivotIndex, rowPivot, costsPivot);
    }

    HANDLE_ERROR(cudaFree(rowPivot));
    HANDLE_ERROR(cudaFree(colPivot));
    return FEASIBLE;
}