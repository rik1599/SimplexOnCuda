#include "twoPhaseMethod.h"
#include "error.cuh"

/** Inserisce una matrice di indentità in coda a una matrice
 * 
 * @param mat - puntatore alla matrice
 * @param rows - righe della matrice
 * @param cols - colonne della matrice
 * @param startCol - colonna di partenza
 * @param dim - dimensione dell'identità
 */
__global__ void fillMatrix(TYPE* mat, int rows, int cols, int startCol, int dim)
{

}

/** Inizializza il vettore della base con numeri progressivi da start a start + size - 1
 */
__global__ void fillBaseVector(int* base, int size, int start)
{

}

/** Implementa in parallelo operazioni del tipo
 * mat[0] = mat[0] - (\sum_{i=i}^{rows-1} coefficient[i]*mat[i])
 * 
 * In pratica sottrae al vettore a riga 0 del tableau (valori della funzione obiettivo)
 * TUTTE le altre righe, ognuna opportunatamente moltiplicata per un coefficiente.
 * 
 * L'operazione viene fatta "per tile"
 * 
 * Si tratta di una variante dell'esercizio dell'istogramma visto a lezione 
 */
__global__ void gaussianElimination(TYPE* mat, TYPE* lastCol, TYPE* coefficients, int rows, int cols)
{

}

/** Scandisce il vettore della base per creare il vettore dei coefficienti.
 * 
 * Sostanzialmente coefficients[i] = firstRow[base[i]]
 */
__global__ void createCoefficientVector(TYPE* firstRow, int cols, int* base, int rows, TYPE* coefficients)
{

}

/** Esprimo la funzione obiettivo in termini delle variabili non di base (vedi es. istogramma)
 */
void updateObjectiveFunction(tabular_t* tabular)
{

}

/**
 * Passaggi per la fase 1 
 * Dati m il numero di vincoli e n il numero di variabili
 * 1) Riempimento tabella (su cuda stream diversi)
 *      1) primi n+m valori della prima riga a 0 (cudaMemsetAsync)
 *      2) ultimi m valori della prima riga a 1 (kernel)
 *      3) copia della matrice dei vincoli originale dalla prima riga di tabular->table sulle prime n colonne (cudaMemcpy2DAsync)
 *      4) riempimento delle successive m colonne con identità (kernel)
 *      5) riempimento ultime m colonne con identità (kernel)
 *      6) copia del vettore dei termini noti nel vettore degli indicatori a partire dall'indice 1 (cudaMemcpyAsync)
 *      7) riempimento vettore della base con numeri progressivi da n+m a n+2m-1 (kernel)
 * 2) Esprimo la funzione obiettivo in termini delle variabili non di base (vedi es. istogramma)
 * 3) Eseguo l'algoritmo di soluzione fino all'ottimo (file a parte)
 * 4) Controllo il primo valore del vettore degli indicatori: se < 0 problema infeasible
 * 5) Se è presente in base un valore x tale che n+m <= x < n+2m, il problema è degenere (fare su un kernel?)
 * 6) Proseguo a fase 2
 */
int phase1(tabular_t* tabular, int* base)
{
    return FEASIBLE;
}

/**
 * Passaggi per la fase 2
 * Dati m il numero di vincoli e n il numero di variabili
 * 
 * 1) Riduco il numero di colonne da n+2m a n+m (basta aggiornare il valore in tabular->cols)
 * 2) Uso due stream per riempire la prima riga
 *      1) Copio il vettore dei coefficienti nei primi n elementi della prima riga (cudaMemcpyAsync)
 *      2) Imposto i restanti m elementi a 0 (cudaMemsetAsync)
 * 3) Esprimo la funzione obiettivo in termini delle variabili non di base (vedi es. istogramma)
 * 4) Eseguo l'algoritmo di soluzione fino all'ottimo (file a parte)
 */
int phase2(tabular_t* tabular, int* base)
{
    return FEASIBLE;
}

__inline__ void unregisterMemory(int* base_h, problem_t* problem)
{
    HANDLE_ERROR(cudaHostUnregister(problem->constraintsMatrix));
    HANDLE_ERROR(cudaHostUnregister(problem->knownTermsVector));
    HANDLE_ERROR(cudaHostUnregister(problem->objectiveFunction));
    HANDLE_ERROR(cudaFreeHost(base_h));
}

int twoPhaseMethod(problem_t* problem, TYPE* solution, TYPE* optimalValue)
{
    tabular_t* tabular = newTabular(problem);
    int* base_h;
    int* base_map;

    // Uso memoria di tipo mapped per memorizzare il vettore di base
    HANDLE_ERROR(cudaHostAlloc(&base_h, tabular->rows * sizeof(int), cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostGetDevicePointer(&base_map, base_h, 0));

    //Registro i vettori del problema come memoria page-locked (per poter utilizzare i trasferimenti paralleli con gli stream)
    HANDLE_ERROR(cudaHostRegister(problem->constraintsMatrix, BYTE_SIZE(problem->vars * problem->constraints), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->knownTermsVector, BYTE_SIZE(problem->constraints), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->objectiveFunction, BYTE_SIZE(problem->vars), cudaHostRegisterDefault));

    int result = phase1(tabular, base_map);
    if (result != FEASIBLE)
    {
        unregisterMemory(base_h, problem);
        return result;
    }
    
    result = phase2(tabular, base_map);
    if (result != FEASIBLE)
    {
        unregisterMemory(base_h, problem);
        return result;
    }
    
    /* Estrai soluzione dalla tabella (si può fare parallelo?)
    Per estrarre la soluzione si prende:
    1) tabular->indicatorCol[0] -> valore ottimale della funzione obiettivo
    2) Per ogni variabile di base: solution[base[i]] = tabular->indicatorCol[i], il resto è 0
    */

    unregisterMemory(base_h, problem);
    return result;
}