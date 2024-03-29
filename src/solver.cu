#include "solver.h"
#include "twoPhaseMethod.h"
#include "reduction.cuh"
#include "error.cuh"

#ifdef TIMER
#include "chrono.cuh"
#endif

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
        dst[i] = *INDEX(matInfo.mat, i, colToCpy, matInfo.pitch);
    }
}

__global__ void updateContraintsMatrix(matrixInfo matInfo, double *colPivot, double *rowPivot, int colPivotIndex, double pivot)
{
    double *pRow;
    char *pMat = (char *)matInfo.mat;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < matInfo.cols; col += blockDim.x * gridDim.x)
    {
        for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < matInfo.rows; row += blockDim.y * gridDim.y)
        {
            pRow = (double *)(pMat + row * matInfo.pitch);
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

__inline__ void updateTableau(tabular_t *tabular, TYPE *colPivot, int colPivotIndex, TYPE *rowPivot, TYPE minCosts, cudaStream_t* streams)
{
    matrixInfo matInfo = {tabular->table, tabular->pitch, tabular->rows, tabular->cols};

    copyColumn<<<BL(tabular->rows), THREADS>>>(matInfo, colPivotIndex, colPivot);
    HANDLE_KERNEL_ERROR();

    TYPE pivot;
    HANDLE_ERROR(cudaMemcpy(&pivot, rowPivot + colPivotIndex, BYTE_SIZE(1), cudaMemcpyDefault));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(BLOCK_DIM(tabular->cols), BLOCK_DIM(tabular->rows));
    updateContraintsMatrix<<<grid, block, 0, streams[0]>>>(matInfo, colPivot, rowPivot, colPivotIndex, pivot);

    updateCostsVector<<<BL(tabular->rows), THREADS, 0, streams[1]>>>(tabular->costsVector, tabular->rows, colPivot, minCosts, pivot);

    HANDLE_KERNEL_ERROR();
}

#define NOT_ENDED -10
__inline__ int solve(tabular_t *tabular, int *base, TYPE *rowPivot, TYPE *colPivot, cudaStream_t* streams)
{
    unsigned int colPivotIndex;
    unsigned int rowPivotIndex;
    TYPE minCosts;

#ifdef TIMER
    start(tabular, "solve");
#endif
    minCosts = minElement(tabular->costsVector + 1, tabular->rows - 1, &rowPivotIndex);
    if (compare(minCosts) < 0)
    {
        HANDLE_ERROR(cudaMemcpy(
            rowPivot,
            ROW(tabular->constraintsMatrix, rowPivotIndex, tabular->pitch),
            BYTE_SIZE(tabular->cols),
            cudaMemcpyDefault));

        if (isLessOrEqualThanZero(rowPivot, tabular->cols))
        {
#ifdef TIMER
            stop();
#endif
            return UNBOUNDED;
        }

        minElement(tabular->knownTermsVector, rowPivot, tabular->cols, &colPivotIndex);
        base[colPivotIndex] = rowPivotIndex;

        updateTableau(tabular, colPivot, colPivotIndex, rowPivot, minCosts, streams);

#ifdef TIMER
        stop();
#endif
#ifdef DEBUG
        printTableauToStream(stdout, tabular, base);
        while (getchar() != '\n')
            ;
#endif
        return NOT_ENDED;
    }
    else
    {
#ifdef TIMER
        stop();
#endif
        return FEASIBLE;
    }
}

int solve(tabular_t *tabular, int *base)
{
    TYPE *rowPivot, *colPivot;
    HANDLE_ERROR(cudaMalloc((void **)&rowPivot, BYTE_SIZE(tabular->cols)));
    HANDLE_ERROR(cudaMalloc((void **)&colPivot, BYTE_SIZE(tabular->rows)));

    cudaStream_t streams[2];
    for (size_t i = 0; i < 2; i++)
        HANDLE_ERROR(cudaStreamCreate(streams + i));

    int status = FEASIBLE;
    while ((status = solve(tabular, base, rowPivot, colPivot, streams)) == NOT_ENDED)
        ;

    HANDLE_ERROR(cudaFree(rowPivot));
    HANDLE_ERROR(cudaFree(colPivot));

    for (size_t i = 0; i < 2; i++)
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));

    return status;
}