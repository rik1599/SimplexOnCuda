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
    tabular->rows = 1 + problem->vars + (2 * problem->constraints);

    allocateGlobalMemory(tabular);

    tabular->knownTermsVector = tabular->table;
    tabular->constraintsMatrix = ROW(tabular->table, 1, tabular->pitch);

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
        tabular->table,
        tabular->pitch,
        BYTE_SIZE(tabular->cols),
        tabular->rows,
        cudaMemcpyDeviceToHost
    ));

    HANDLE_ERROR(cudaMemcpy2D(
        hIndicators,
        BYTE_SIZE(tabular->cols),
        tabular->knownTermsVector,
        tabular->pitch,
        BYTE_SIZE(tabular->cols),
        1,
        cudaMemcpyDeviceToHost
    ));

    HANDLE_ERROR(cudaMemcpy(
        hCosts,
        tabular->costsVector,
        BYTE_SIZE(tabular->rows),
        cudaMemcpyDeviceToHost
    ));

    fprintf(Stream, "\n--------------- Tabular --------------\n");
    for (size_t i = 0; i < tabular->rows; i++)
    {
        for (size_t j = 0; j < tabular->cols; j++)
        {
            fprintf(Stream, "%.2lf\t", hTable[i * tabular->cols + j]);
        }

        fprintf(Stream, "\t|\t %.11lf\n", hCosts[i]);
        if(i==0) fprintf(Stream, "\n");
    }
}

void printTableauToStream(FILE* Stream, tabular_t* tabular, int* base)
{
    if (tabular->table != NULL)
    {
        print(Stream, tabular);
        fprintf(Stream, "Base\n");
        for (int i = 0; i < tabular->cols; i++)
        {
            fprintf(Stream, "%d\t", base[i]);
        }
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