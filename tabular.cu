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

}

tabular_t* newTabular(problem_t* problem)
{
    tabular_t* tabular = (tabular_t*)malloc(sizeof(tabular_t));
    
    tabular->problem = problem;
    //Registro i vettori del problema come memoria page-locked (per poter utilizzare i trasferimenti paralleli con gli stream)
    HANDLE_ERROR(cudaHostRegister(problem->constraintsMatrix, BYTE_SIZE(problem->vars * problem->constraints), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->knownTermsVector, BYTE_SIZE(problem->constraints), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->objectiveFunction, BYTE_SIZE(problem->vars), cudaHostRegisterDefault));

    tabular->rows = problem->constraints + 1;
    tabular->cols = problem->constraints + 2 * problem->vars;

    allocateGlobalMemory(tabular);

    return tabular;
}

void print(FILE* Stream, tabular_t* tabular)
{
    TYPE* hTable = (TYPE*)malloc(BYTE_SIZE(tabular->rows * tabular->cols));
    TYPE* hIndicators = (TYPE*)malloc(BYTE_SIZE(tabular->cols));

    HANDLE_ERROR(cudaMemcpy2D(
        hTable,
        BYTE_SIZE(tabular->cols),
        tabular->table,
        tabular->pitch,
        BYTE_SIZE(tabular->cols),
        tabular->rows,
        cudaMemcpyDeviceToHost
    ));

    HANDLE_ERROR(cudaMemcpy(
        hIndicators,
        tabular->indicatorCol,
        tabular->rows,
        cudaMemcpyDeviceToHost
    ));

    fprintf(Stream, "\n--------------- Tabular --------------\n");
    for (size_t i = 0; i < tabular->rows; i++)
    {
        for (size_t j = 0; j < tabular->cols; j++)
        {
            fprintf(Stream, "%.2lf\t", hTable[i * tabular->cols + j]);
        }
        fprintf(Stream, "%.2lf\n", hIndicators[i]);
    }
    fprintf(Stream, "\n--------------------------------------\n");
    fprintf(Stream, "Base: ");
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
    HANDLE_ERROR(cudaHostUnregister(tabular->problem->constraintsMatrix));
    HANDLE_ERROR(cudaHostUnregister(tabular->problem->knownTermsVector));
    HANDLE_ERROR(cudaHostUnregister(tabular->problem->objectiveFunction));

    if (tabular->table != NULL)
    {
        HANDLE_ERROR(cudaFree(tabular->table));
        HANDLE_ERROR(cudaFree(tabular->indicatorCol));
    }

    free(tabular);
}