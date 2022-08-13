#include "twoPhaseMethod.h"
#include "error.cuh"

/** Inserisce due matrici di indentità in coda a una matrice
 *  Si suppone sia linearizzata per colonne (non penso sia possibile generalizzare)
 * 
 * @param mat - puntatore alla matrice
 * @param rows - righe della matrice
 * @param cols - colonne della matrice
 * @param startRow - riga di partenza
 * @param dim - dimensione dell'identità
 * @param pitch - il pitch della matrice
 */
__global__ void fillMatrix(TYPE* mat, int rows, int cols, int startRow, int dim, size_t pitch)
{
    //grid stride per la generalità
    for(int idX = threadIdx.x + blockIdx.x*blockDim.x; idX < dim; idX += gridDim.x * blockDim.x){
        *(INDEX(mat, (idX+startRow), idX, pitch)) = 1;
        *(INDEX(mat, (idX+startRow+dim), idX, pitch)) = 1;
    }
}

/**
 * Inizializza il vettore della base con numeri progressivi da start a start + size - 1
 */
__global__ void fillBaseVector(int* base, int size, int start)
{
    for(int idX = threadIdx.x + blockIdx.x*blockDim.x; idX < size; idX += gridDim.x * blockDim.x){
        base[idX] = (start+idX);
    }
}

/**
 * Setta a 1 tutti gli elementi del vettore da start a size-1
 */
__global__ void setVectorToOne(TYPE* vector, int size, int start)
{
    for(int idX = threadIdx.x + blockIdx.x*blockDim.x; start + idX < size; idX += gridDim.x * blockDim.x){
        vector[idX] = 1;
    }
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
 * 
 * Funziona al meglio se chiamato con blocchi 32x32 e grid di dimensione (1, k);
 * 
 * Ogni blocco esegue su due volte la sua dimensione orizzontale
 * 
 * Da usare per CC < 6.0
 * 
 * @param mat - il puntatore alla matrice dei vincoli
 * @param objectiveFunction - il puntatore alla funzione obiettivo
 * @param coefficients - il puntatore al vettore dei coefficienti
 * @param rows - il numero di righe di mat (se corretto è anche uguale alla dimensione della funzione obiettivo)
 * @param columns - il numero di colonne di mat (se corretto è anche uguale alla dimensione del vettore dei coefficienti)
 */
__global__ void gaussianElimination(TYPE* mat, TYPE* objectiveFunction, TYPE* coefficients, int rows, int cols, size_t pitch)
{
    int colsTotal = ((cols + blockDim.x - 1)/blockDim.x) * blockDim.x;

    for(int idX = threadIdx.x + blockIdx.x * blockDim.x * 2; idX<colsTotal; idX += 2 * gridDim.x*blockDim.x){
        for(int idY = threadIdx.y + blockIdx.y * blockDim.y; idY<rows; idY += gridDim.y*blockDim.y){

            // prendiamo il valore per questo specifico thread sommando nel caricamento (stando attenti a settare a zero il valore)
            TYPE value = idX < cols && idX + blockDim.x < cols ?  *(INDEX(mat, idY, idX, pitch)) * coefficients[idX] +
                                                                  *(INDEX(mat, idY, idX + blockDim.x, pitch)) * coefficients[idX + blockDim.x]
                                                                : (idX < cols ? *(INDEX(mat, idY, idX, pitch)) * coefficients[idX]
                                                                : 0);

            __syncthreads();

            //riduzione intra warp
            for(int offset = 1; offset < 32; offset *= 2){
                value += __shfl_xor_sync(0xffffff, value, offset);
            }

            if(threadIdx.x == 0){
                atomicAdd(&objectiveFunction[idY], -1 * value);
            }
            
        }
    }
}

/**
 * Stesso di quello sopra ed ha prestazioni simili ma è più semplice
 * 
 * Ogni blocco esegue su due volte la sua dimensione orizzontale
 * 
 * Da usare per CC > 6.0
 * 
 * @param mat - il puntatore alla matrice dei vincoli
 * @param objectiveFunction - il puntatore alla funzione obiettivo
 * @param coefficients - il puntatore al vettore dei coefficienti
 * @param rows - il numero di righe di mat (se corretto è anche uguale alla dimensione della funzione obiettivo)
 * @param columns - il numero di colonne di mat (se corretto è anche uguale alla dimensione del vettore dei coefficienti)
 */
__global__ void gaussianEliminationNaive(TYPE* mat, TYPE* objectiveFunction, TYPE* coefficients, int rows, int cols, size_t pitch){
    for(int idX = threadIdx.x + (blockIdx.x * blockDim.x * 2); idX<cols; idX += 2 * gridDim.x*blockDim.x){
        for(int idY = threadIdx.y + blockIdx.y * blockDim.y; idY<rows; idY += gridDim.y*blockDim.y){    
            atomicAdd(&objectiveFunction[idY], -1 * (idX + blockDim.x < cols ? ((*(INDEX(mat, idY, idX, pitch)) * coefficients[idX]) +
                                                                            (*(INDEX(mat, idY, (idX + blockDim.x), pitch))) * coefficients[idX + blockDim.x]) : 
                                                                            (*(INDEX(mat, idY, idX, pitch))) * coefficients[idX]));
        }
    }
}


/** Scandisce il vettore della base per creare il vettore dei coefficienti.
 * 
 * Sostanzialmente coefficients[i] = firstRow[base[i]]
 * 
 * @param firstRow - puntatore al primo elemento della funzione obiettivo (quindi funzione obiettivo + 1)
 * @param baseSize - dimensione del vettore della base e del vettore dei coefficienti
 * @param base - punteatore al vettore della base
 * @param coefficients - puntatore al vettore dove salvare i coefficienti
 */
__global__ void createCoefficientVector(TYPE* firstRow, int baseSize, int* base, TYPE* coefficients)
{
    for(int idX = threadIdx.x + blockIdx.x * blockDim.x; idX<baseSize; idX += gridDim.x*blockDim.x){
        coefficients[idX] = firstRow[base[idX]]; 
    } 
}

/**
 * Esprimo la funzione obiettivo in termini delle variabili non di base (vedi es. istogramma)
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