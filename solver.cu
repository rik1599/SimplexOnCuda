#include "solver.h"
#include "error.cuh"

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

}

/**
 * Passaggi per la fase 2
 * Dati m il numero di vincoli e n il numero di variabili
 * 
 * 1) Riduco il numero di colonne da n+2m a n+m (basta aggiornare il valore in tabular->cols)
 * 2) Uso due stream per riempire la prima riga
 *      1) Copio il vettore dei coefficienti nei primi n elementi della prima riga (cudaMemcpyAsync)
 *      2) Imposto i restanti m elementi a 0 (cudaMemsetAsync)
 * 
 */
int phase2(tabular_t* tabular, int* base)
{

}

int solve(problem_t* problem, TYPE* solution, TYPE* optimalValue)
{
    tabular_t* tabular = newTabular(problem);
    int* base_h;
    int* base_map;

    // Uso memoria di tipo mapped per memorizzare il vettore di base
    HANDLE_ERROR(cudaHostAlloc(&base_h, tabular->rows * sizeof(int), cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostGetDevicePointer(&base_map, base_h, 0));

    int result = phase1(tabular, base_map);
    if (result == INFEASIBLE || result == DEGENERATE)
    {
        cudaFreeHost(base_h);
        return result;
    }
    
    result = phase2(tabular, base_map);
    if (result == UNBOUNDED)
    {
        cudaFreeHost(base_h);
        return result;
    }
    
    /* Estrai soluzione dalla tabella (si può fare parallelo?)
    Per estrarre la soluzione si prende:
    1) tabular->indicatorCol[0] -> valore ottimale della funzione obiettivo
    2) Per ogni variabile di base: solution[base[i]] = tabular->indicatorCol[i], il resto è 0
    */

    cudaFreeHost(base_h);
    return result;
}