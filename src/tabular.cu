#include "tabular.cuh"
#include "error.cuh"

/**
 * Allocazione spazio in memoria globale
 * Dati m il numero di vincoli e n il numero di variabili
 * 1) Allocazione matrice di dimensioni (m+1)*(n+2m) (cudaMallocPitch)
 * 3) Allocazione vettore indicatori di dimensione m+1
 */
void allocateGlobalMemory(tabular_t* tabular)
{
    HANDLE_ERROR(cudaMallocPitch(
        (void**)&tabular->table,
        &tabular->pitch,
        BYTE_SIZE(tabular->cols),
        tabular->rows
    ));

    HANDLE_ERROR(cudaMalloc(
        (void**)&tabular->costsVector,
        BYTE_SIZE(tabular->rows)
    ));
}

tabular_t* newTabular(problem_t* problem)
{
    tabular_t* tabular = (tabular_t*)malloc(sizeof(tabular_t));
    
    tabular->problem = problem;
    tabular->cols = problem->constraints;
    tabular->rows = (problem->vars + 1) + 2 * problem->constraints;

    allocateGlobalMemory(tabular);

    tabular->indicatorsVector = tabular->table;
    tabular->constraintsMatrix = (TYPE*)((char*)tabular->table + tabular->pitch);

    return tabular;
}

__inline__ void print(FILE* Stream, tabular_t* tabular)
{
    TYPE* hTable = (TYPE*)malloc(BYTE_SIZE(tabular->rows * tabular->cols));
    TYPE* hIndicators = (TYPE*)malloc(BYTE_SIZE(tabular->cols));
    TYPE* hCosts = (TYPE*)malloc(BYTE_SIZE(tabular->rows));

    HANDLE_ERROR(cudaMemcpy2D(
        hTable,
        BYTE_SIZE(tabular->cols),
        tabular->constraintsMatrix,
        tabular->pitch,
        BYTE_SIZE(tabular->cols),
        tabular->rows - 1,
        cudaMemcpyDeviceToHost
    ));

    HANDLE_ERROR(cudaMemcpy2D(
        hIndicators,
        BYTE_SIZE(tabular->cols),
        tabular->indicatorsVector,
        tabular->pitch,
        BYTE_SIZE(tabular->cols),
        1,
        cudaMemcpyDeviceToHost
    ));

    HANDLE_ERROR(cudaMemcpy(
        hCosts,
        tabular->costsVector,
        tabular->rows,
        cudaMemcpyDeviceToHost
    ));

    fprintf(Stream, "\n--------------- Tabular --------------\n");
    for (size_t i = 0; i < tabular->rows-1; i++)
    {
        for (size_t j = 0; j < tabular->cols; j++)
        {
            fprintf(Stream, "%.2lf\t", hTable[i * tabular->cols + j]);
        }
        fprintf(Stream, "%.2lf\n", hIndicators[i]);
    }
    fprintf(Stream, "\n--------------------------------------\n");
    fprintf(Stream, "Vettore dei costi: ");
    for (size_t i = 0; i < tabular->rows; i++)
    {
        fprintf(Stream, "%.2lf\t", hCosts[i]);
    }
}

void printTableauToStream(FILE* Stream, tabular_t* tabular)
{
    if (tabular->table != NULL)
    {
        print(Stream, tabular);
    }
}

void freeTabular(tabular_t* tabular)
{
    if (tabular->table != NULL)
    {
        HANDLE_ERROR(cudaFree(tabular->table));
        HANDLE_ERROR(cudaFree(tabular->costsVector));
    }

    free(tabular);
}