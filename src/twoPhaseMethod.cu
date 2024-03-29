#include "twoPhaseMethod.h"
#include "error.cuh"
#include "gaussian.cuh"

#ifdef TIMER
#include "chrono.cuh"
#endif

#define THREADS 512
#define BL(N) min((N + THREADS - 1) / THREADS, 1024)

#ifdef TIMER
    bool benchmark = false;
    void enableBenchmarkMode(){
        benchmark = true;
    }

    void disableBenchmarkMode(){
        benchmark = false;
    }
#endif

/** Inserisce due matrici di indentità in coda a una matrice
 *  Si suppone sia linearizzata per colonne (non penso sia possibile generalizzare)
 *
 * @param mat - puntatore alla matrice
 * @param cols - colonne della matrice
 * @param pitch - il pitch della matrice
 */
__global__ void fillMatrix(TYPE *mat, int cols, size_t pitch)
{
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x;
         idX < cols;
         idX += gridDim.x * blockDim.x)
    {
        *(INDEX(mat, idX, idX, pitch)) = 1;
        *(INDEX(mat, idX + cols, idX, pitch)) = 1;
    }
}

/**
 * Inizializza il vettore della base con numeri progressivi da start a start + size - 1
 */
__global__ void fillBaseVector(int *base, int size, int start)
{
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x;
         idX < size;
         idX += gridDim.x * blockDim.x)
    {
        base[idX] = (start + idX);
    }
}

/**
 * Setta a 1 tutti gli elementi del vettore da start alla fine del vettore
 *
 * @param vector - puntatore al vettore da settare ad 1
 * @param size - la dimensione del vettore
 */
__global__ void setVectorToOne(TYPE *vector, int size)
{
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x;
         idX < size;
         idX += gridDim.x * blockDim.x)
    {
        vector[idX] = 1;
    }
}

/**
 * Inverte i segni a tutti gli elementi di un vettore
 *
 * @param vector - puntatore al vettore
 * @param size - la dimensione del vettore
 */
__global__ void negateVector(TYPE *vector, int size)
{
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x;
         idX < size;
         idX += gridDim.x * blockDim.x)
    {
        vector[idX] = -vector[idX];
    }
}

__global__ void negateColumn(TYPE *mat, int height, size_t pitch, int colIndex)
{
    TYPE *pValue = NULL;
    TYPE value = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < height;
         i += blockDim.x * gridDim.x)
    {
        pValue = INDEX(mat, i, colIndex, pitch);
        value = *pValue;
        *pValue = -value;
    }
}

__global__ void checkColumns(TYPE *mat, int width, int height, size_t pitch)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < width;
         i += blockDim.x * gridDim.x)
    {
        if (compare(*INDEX(mat, 0, i, pitch)) < 0)
        {
            negateColumn<<<BL(height), THREADS>>>(mat, height, pitch, i);
        }
    }
}

/*
 * Genera in parallelo il vettore della soluzione nella memoria device
 */
__global__ void getSolution(TYPE *source, int *base, int baseSize, TYPE *out, int lastVar)
{
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x;
         idX < baseSize;
         idX += gridDim.x * blockDim.x)
    {
        int var = base[idX];
        if (var < lastVar)
        {
            out[var] = source[idX];
        }
    }
}

/*
 * Dato un valore minimo ed un valore massimo controlla se nel vettore in input ce ne è uno compreso
 * (min <= x < max)
 */
__global__ void countElementsInRange(int *vector, int size, int min, int max, unsigned int *out)
{
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x;
         idX < size;
         idX += gridDim.x * blockDim.x)
    {
        if (vector[idX] < max && vector[idX] >= min)
            atomicAdd(out, 1);
    }
}

void fillTableu(tabular_t *tabular, int *base)
{
    cudaStream_t streams[6];
    for (size_t i = 0; i < 6; i++)
        HANDLE_ERROR(cudaStreamCreate(streams + i));

    // Punto 1: primi n + m valori a 0 della funzione dei costi
    int sizeToSetZero = tabular->problem->vars + tabular->problem->constraints + 1;
    HANDLE_ERROR(cudaMemsetAsync(tabular->costsVector, 0, BYTE_SIZE(sizeToSetZero), streams[0]));

    // Punto 2: ultimi m valori della prima riga a 1 (kernel)
    setVectorToOne<<<BL(tabular->problem->constraints), THREADS, 0, streams[1]>>>
        (tabular->costsVector + sizeToSetZero, tabular->problem->constraints);
    
    // Punto 3: copia della matrice dei vincoli originale dalla seconda riga di tabular->table sulle prime n colonne (cudaMemcpy2DAsync)
    HANDLE_ERROR(cudaMemcpy2DAsync(
        tabular->constraintsMatrix,               // destinazione
        tabular->pitch,                           // pitch della destinazione
        tabular->problem->constraintsMatrix,      // fonte
        BYTE_SIZE(tabular->problem->constraints), // pitch della fonte
        BYTE_SIZE(tabular->problem->constraints), // larghezza matrice
        tabular->problem->vars,                   // altezza matrice
        cudaMemcpyDefault,                        // tipo
        streams[2]                                // stream
        ));

    // Punto 4 e 5: riempimento delle successive m colonne con identità (kernel)
    fillMatrix<<<BL(tabular->cols), THREADS, 0, streams[3]>>>(
        ROW(tabular->constraintsMatrix, tabular->problem->vars, tabular->pitch),
        tabular->cols, // colonne della matrice
        tabular->pitch // pitch
    );

    // Punto 6: copia del vettore dei termini noti nel vettore degli indicatori (cudaMemcpyAsync)
    HANDLE_ERROR(cudaMemcpyAsync(
        tabular->knownTermsVector,                // puntatore al vettore destinazione (vettore indicatori)
        tabular->problem->knownTermsVector,       // fonte
        BYTE_SIZE(tabular->problem->constraints), // dimensione in byte del vettore
        cudaMemcpyDefault,                        // tipo
        streams[4]                                // stream
        ));

    // Punto 7: riempimento vettore della base con numeri progressivi da n+m a n+2m-1 (kernel)
    fillBaseVector<<<BL(tabular->cols), THREADS, 0, streams[5]>>>(
        base,
        tabular->cols,
        tabular->problem->vars + tabular->problem->constraints);

    HANDLE_KERNEL_ERROR();

    for (size_t i = 0; i < 6; i++)
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));

    // Punto 8: negare tutte le disequazioni (colonne del tableau) con termine noto < 0
    checkColumns<<<BL(tabular->cols), THREADS>>>(tabular->table, tabular->cols, tabular->rows, tabular->pitch);
}

/**
 * Controlla se il problema è degenere
 * @return DEGENERATE se degenere, FEASIBLE altrimenti
 */
int checkDegeneracy(int *base, int size, int firstArtificial, int endArtificial)
{
    unsigned int *out;
    HANDLE_ERROR(cudaMalloc((void **)&out, sizeof(unsigned int)));
    HANDLE_ERROR(cudaMemset(out, 0, sizeof(unsigned int)));

    countElementsInRange<<<BL(size), THREADS>>>(base, size, firstArtificial, endArtificial, out);

    int result;
    HANDLE_ERROR(cudaMemcpy(&result, out, sizeof(unsigned int), cudaMemcpyDefault));

    HANDLE_ERROR(cudaFree(out));

    if (result > 0)
        return DEGENERATE;
    else
        return FEASIBLE;
}

int phase1(tabular_t *tabular, int *base_h, int *base_dev)
{
    // Fase 1: riempimento del tableu
    printf("Phase 1: Filling Tableau\n");
#ifdef TIMER
    start(tabular, "fillTableau");
#endif
    fillTableu(tabular, base_dev);
#ifdef TIMER
    stop();
#endif

#ifdef DEBUG
    fprintf(stdout, "\nTableu nella situazione iniziale\n");
    printTableauToStream(stdout, tabular, base_h);
#endif

    // Fase 2: eliminazione di gauss
    printf("Phase 1: Resetting out-of-base variables\n");
#ifdef TIMER
    start(tabular, "gauss1");
#endif
    updateObjectiveFunction(tabular, base_dev);
#ifdef TIMER
    stop();
#endif
#ifdef DEBUG
    fprintf(stdout, "\nTableu dopo l'eliminazione di gauss\n");
    printTableauToStream(stdout, tabular, base_h);
#endif

    // Fase 3: lancio del solver
    printf("Phase 1: Solving auxiliary problem\n");
    solve(tabular, base_h);
#ifdef DEBUG
    fprintf(stdout, "\nTableu dopo il lancio del primo solver\n");
    printTableauToStream(stdout, tabular, base_h);
#endif

    // Fase 4: controllo infattibilità
    TYPE firstKnownTermsValue;
    HANDLE_ERROR(cudaMemcpy(&firstKnownTermsValue, tabular->costsVector, BYTE_SIZE(1), cudaMemcpyDeviceToHost));
    if (compare(firstKnownTermsValue) < 0)
        return INFEASIBLE;

        // Fase 5: controllo degenere: se è presente in base un valore x tale che n+m <= x < n+2m, il problema è degenere
#ifdef TIMER
    start(tabular, "checkDegeneracy");
#endif
    int isDegenerate = checkDegeneracy(
        base_dev,
        tabular->cols,
        tabular->problem->vars + tabular->problem->constraints,
        tabular->problem->vars + 2 * tabular->problem->constraints);
#ifdef TIMER
    stop();
#endif
    return isDegenerate;
}

int phase2(tabular_t *tabular, int *base_h, int *base_dev)
{
    // Fase 1: riduzione del numero di colonne
    tabular->rows -= tabular->cols;

#ifdef DEBUG
    fprintf(stdout, "\nTableu dopo aggiornamento colonne in phase2\n");
    printTableauToStream(stdout, tabular, base_h);
#endif

    // Fase 2: riempimento vettore costi su due stream diversi
    printf("Phase 2: Filling costs vector with the original one\n");

    cudaStream_t streams[2];
    for (size_t i = 0; i < 2; i++)
        HANDLE_ERROR(cudaStreamCreate(streams + i));

#ifdef TIMER
    start(tabular, "costsVector");
#endif
    // ultimi m elementi a 0
    HANDLE_ERROR(cudaMemsetAsync(
        1 + tabular->costsVector + tabular->problem->vars,
        0,
        BYTE_SIZE(tabular->cols),
        streams[0]));

    HANDLE_ERROR(cudaMemcpyAsync(
        tabular->costsVector + 1,
        tabular->problem->objectiveFunction,
        BYTE_SIZE(tabular->problem->vars),
        cudaMemcpyDefault,
        streams[1]));
    negateVector<<<BL(tabular->problem->vars), THREADS, 0, streams[1]>>>(tabular->costsVector + 1, tabular->problem->vars);
    HANDLE_KERNEL_ERROR();
#ifdef TIMER
    stop();
#endif

    for (size_t i = 0; i < 2; i++)
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));

#ifdef DEBUG
    fprintf(stdout, "\nTableu dopo riempimento funzione obiettivo in phase2\n");
    printTableauToStream(stdout, tabular, base_h);
#endif

    // Fase 3: Eliminazione di gauss per esprimere la funzione obiettivo in termini delle variabili non di base
    printf("Phase 2: Resetting out-of-base variables\n");
#ifdef TIMER
    start(tabular, "gauss2");
#endif
    updateObjectiveFunction(tabular, base_dev);
#ifdef DEBUG
    fprintf(stdout, "\nTableu dopo eliminazione di gauss in phase2\n");
    printTableauToStream(stdout, tabular, base_h);
#endif
#ifdef TIMER
    stop();
#endif

    // Fase 4: Esecuzione dell'algoritmo di risoluzione
    printf("Phase 2: Solving original problem\n");
#ifdef DEBUG
    int esito = solve(tabular, base_h);
    fprintf(stdout, "\nTableu dopo seconda esecuzione del solver\n");
    printTableauToStream(stdout, tabular, base_h);
    return esito;
#else
    return solve(tabular, base_h);
#endif
}

__inline__ void unregisterMemory(int *base_h, problem_t *problem)
{
    HANDLE_ERROR(cudaHostUnregister(problem->constraintsMatrix));
    HANDLE_ERROR(cudaHostUnregister(problem->knownTermsVector));
    HANDLE_ERROR(cudaHostUnregister(problem->objectiveFunction));
    HANDLE_ERROR(cudaFreeHost(base_h));

#ifdef TIMER
    closeCsv();
#endif
}

void getSolutionHost(tabular_t *tabular, int *base, TYPE *solution, TYPE *optimalValue)
{
    HANDLE_ERROR(cudaMemcpy(optimalValue, tabular->costsVector, BYTE_SIZE(1), cudaMemcpyDefault));

    TYPE *dev_solution;
    HANDLE_ERROR(cudaHostRegister(solution, BYTE_SIZE(tabular->problem->vars), cudaHostRegisterMapped));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_solution, solution, 0));
    HANDLE_ERROR(cudaMemset(dev_solution, 0, BYTE_SIZE(tabular->problem->vars)));

    getSolution<<<BL(tabular->cols), THREADS>>>(tabular->knownTermsVector, base, tabular->cols, dev_solution, tabular->problem->vars);
    HANDLE_KERNEL_ERROR();

    HANDLE_ERROR(cudaHostUnregister(solution));
}

int twoPhaseMethod(problem_t *problem, TYPE *solution, TYPE *optimalValue)
{
    tabular_t *tabular = newTabular(problem);
    int *base_h;
    int *base_map;

    // Uso memoria di tipo mapped per memorizzare il vettore di base
    HANDLE_ERROR(cudaHostAlloc(&base_h, tabular->cols * sizeof(int), cudaHostAllocMapped)); // il vettore della base ha dimensione vettore dei vincoli => tabular->cols
    HANDLE_ERROR(cudaHostGetDevicePointer(&base_map, base_h, 0));

    // Registro i vettori del problema come memoria page-locked (per poter utilizzare i trasferimenti paralleli con gli stream)
    HANDLE_ERROR(cudaHostRegister(problem->constraintsMatrix, BYTE_SIZE(problem->vars * problem->constraints), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->knownTermsVector, BYTE_SIZE(problem->constraints), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->objectiveFunction, BYTE_SIZE(problem->vars), cudaHostRegisterDefault));

#ifdef TIMER
    if(benchmark){
        initCsvBenchmark(problem->vars, problem->constraints);
    }else{
        initCsv();
    }
#endif

    int result = phase1(tabular, base_h, base_map);
    if (result != FEASIBLE)
    {
        unregisterMemory(base_h, problem);
        freeTabular(tabular);
        return result;
    }

    result = phase2(tabular, base_h, base_map);
    if (result != FEASIBLE)
    {
        unregisterMemory(base_h, problem);
        freeTabular(tabular);
        return result;
    }

#ifdef TIMER
    start(tabular, "solution");
#endif
    getSolutionHost(tabular, base_map, solution, optimalValue);
#ifdef TIMER
    stop();
#endif

    unregisterMemory(base_h, problem);
    freeTabular(tabular);
    return result;
}