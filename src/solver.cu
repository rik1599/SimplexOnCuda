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
#define BLOCK_DIM(N) ceil((N + 512.0) / 544.0)

#define THREADS 512
#define BL(N) min((N + THREADS - 1) / THREADS, 1024)

__global__ void copyColumn(matrixInfo matInfo, int colToCpy, TYPE *dst)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < matInfo.rows;
         i += blockDim.x * gridDim.x)
    {
        dst[i] = ROW(matInfo.mat, i, matInfo.pitch)[colToCpy];
    }
}

__global__ void updateVariables(matrixInfo matInfo, double *colPivot, double *rowPivot, int colPivotIndex, double pivot)
{
    // coordinate
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // dimensioni griglia
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    double *pRow;
    for (int col = x; col < matInfo.cols; col += nx)
    {
        for (int row = y; row < matInfo.rows; row += ny)
        {
            pRow = ROW(matInfo.mat, row, matInfo.pitch);
            pRow[col] = col == colPivotIndex ? pRow[col] / pivot : fma(-rowPivot[col] / pivot, colPivot[row], pRow[col]);
        }
    }
}

__global__ void updateCostsVector(TYPE *costVector, int size, double *colPivot, double costsPivot, double pivot)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x)
    {
        costVector[i] = fma(-costsPivot / pivot, colPivot[i], costVector[i]);
    }
}

__inline__ void updateAll(tabular_t *tabular, TYPE *colPivot, int colPivotIndex, TYPE *rowPivot, TYPE minCosts)
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
    updateCostsVector<<<BL(tabular->rows), THREADS, 0, streams[1]>>>(tabular->costsVector, tabular->rows, colPivot, minCosts, pivot);

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

    TYPE minCosts = minElement(tabular->costsVector + 1, tabular->rows - 1, &rowPivotIndex);
    while (compare(minCosts) < 0)
    {
        HANDLE_ERROR(cudaMemcpy(rowPivot, ROW(tabular->constraintsMatrix, rowPivotIndex, tabular->pitch), BYTE_SIZE(tabular->cols), cudaMemcpyDefault));

        if (isLessThanZero(rowPivot, tabular->cols))
        {
            return UNBOUNDED;
        }

        minElement(tabular->knownTermsVector, rowPivot, tabular->cols, &colPivotIndex);
        base[colPivotIndex] = rowPivotIndex;

        updateAll(tabular, colPivot, colPivotIndex, rowPivot, minCosts);
#ifdef DEBUG
        printTableauToStream(stdout, tabular, base);
        while (getchar() != '\n')
            ;
#endif
        minCosts = minElement(tabular->costsVector + 1, tabular->rows - 1, &rowPivotIndex);
    }

    HANDLE_ERROR(cudaFree(rowPivot));
    HANDLE_ERROR(cudaFree(colPivot));
    return FEASIBLE;
}