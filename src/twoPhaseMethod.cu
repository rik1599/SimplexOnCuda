#include "twoPhaseMethod.h"
#include "error.cuh"


/** Inserisce due matrici di indentità in coda a una matrice
 *  Si suppone sia linearizzata per colonne (non penso sia possibile generalizzare)
 *
 * @param mat - puntatore alla matrice
 * @param cols - colonne della matrice
 * @param pitch - il pitch della matrice
 */
__global__ void fillMatrix(TYPE *mat, int cols, size_t pitch)
{
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;

    // grid stride per la generalità
    for (; idX < cols; idX += step)
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
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x; idX < size; idX += gridDim.x * blockDim.x)
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
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    for (; idX < size; idX += step)
    {
        vector[idX] = 1;
    }
}

/**
 * Inverte i segni a tutti gli elementi del vettore da start alla fine del vettore
 *
 * @param vector - puntatore al vettore da settare ad 1
 * @param size - la dimensione del vettore
 */
__global__ void negateVector(TYPE *vector, int size)
{
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    for (; idX < size; idX += step)
    {
        vector[idX] = -vector[idX];
    }
}

/**
 * Carica due elementi dalla matrice a distanza un blocco:
 * -se puo caricarne 2 li somma
 * -se può caricarne 1 lo carica e basta
 * -se non può caricarne nessuno ritorna 0 
 */
__inline__ __device__ TYPE loadValueWithPreSumZero(TYPE* mat, TYPE* coefficients, int idX, int idY, size_t pitch, int rows, int cols){
    return (idX + blockDim.x < cols ?
                                      *(INDEX(mat, idY, idX, pitch)) * coefficients[idX]                                    //doppio caaricamento
                                            + *(INDEX(mat, idY, idX + blockDim.x, pitch)) * coefficients[idX + blockDim.x]  
                                    : (idX < cols ?
                                                    *(INDEX(mat, idY, idX, pitch)) * coefficients[idX]                      //caricamento singolo
                                                  : 0));                                                                    //zero
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
__global__ void gaussianElimination(TYPE *mat, TYPE *objectiveFunction, TYPE *coefficients, int rows, int cols, size_t pitch)
{
    int colsTotal = ((cols + blockDim.x - 1) / blockDim.x) * blockDim.x;

    for (int idX = threadIdx.x + blockIdx.x * blockDim.x * 2; idX < colsTotal; idX += 2 * gridDim.x * blockDim.x)
    {
        for (int idY = threadIdx.y + blockIdx.y * blockDim.y; idY < rows; idY += gridDim.y * blockDim.y)
        {

            // prendiamo il valore per questo specifico thread sommando nel caricamento (stando attenti a settare a zero il valore)
            TYPE value = loadValueWithPreSumZero(mat, coefficients, idX, idY, pitch, rows, cols);
            __syncthreads();

            // riduzione intra warp
            for (int offset = 1; offset < 32; offset *= 2)
            {
                value += __shfl_xor_sync(0xffffff, value, offset);
            }

            if (threadIdx.x == 0)
            {
                atomicAdd(&objectiveFunction[idY], -value);
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
__global__ void gaussianEliminationNaive(TYPE *mat, TYPE *objectiveFunction, TYPE *coefficients, int rows, int cols, size_t pitch)
{
    for (int idX = threadIdx.x + (blockIdx.x * blockDim.x * 2); idX < cols; idX += 2 * gridDim.x * blockDim.x)
    {
        for (int idY = threadIdx.y + blockIdx.y * blockDim.y; idY < rows; idY += gridDim.y * blockDim.y)
        {
            atomicAdd(&objectiveFunction[idY], -loadValueWithPreSumZero(mat, coefficients, idX, idY, pitch, rows, cols));
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
__global__ void createCoefficientVector(TYPE *firstRow, int baseSize, int *base, TYPE *coefficients)
{
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;

    for (; idX < baseSize; idX += step)
    {
        coefficients[idX] = firstRow[base[idX]];
    }
}

/*
 * Genera in parallelo il vettore della soluzione nella memoria device
 */
__global__ void getSolution(TYPE *source, int *base, int baseSize, TYPE *out, int lastVar)
{
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x; idX < baseSize; idX += gridDim.x * blockDim.x)
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
__global__ void checkVector(int *vector, int size, int min, int max, int *out)
{
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x; idX < size; idX += gridDim.x * blockDim.x)
    {
        if (vector[idX] < max && vector[idX] >= min)
        {
            atomicAdd(out, 1);
        }
    }
}

/**
 * Esprimo la funzione obiettivo in termini delle variabili non di base (vedi es. istogramma)
 */
void updateObjectiveFunction(tabular_t *tabular, int *base)
{
    // per prima cosa dobbiamo creare il vettore dei coefficienti
    TYPE *coefficientVector;
    HANDLE_ERROR(cudaMalloc((void **)&coefficientVector, BYTE_SIZE(tabular->cols)));

    createCoefficientVector<<<BL(tabular->cols), THREADS>>>(
        tabular->costsVector + 1,
        tabular->cols,
        base,
        coefficientVector);
    HANDLE_KERNEL_ERROR();

    // adesso abbiamo il vettore dei coefficienti, possiamo eseguire l'algoritmo di gauss

    dim3 block = dim3(32, 32);
    dim3 grid = dim3(1,
                     (tabular->rows + (block.y - 1)) / block.y);

    gaussianElimination<<<grid, block>>>(tabular->table,
                                         tabular->costsVector,
                                         coefficientVector,
                                         tabular->rows,
                                         tabular->cols,
                                         tabular->pitch);
    HANDLE_KERNEL_ERROR();

    HANDLE_ERROR(cudaFree(coefficientVector));
}

/**
 * Riempimento del tableu su cuda stream diversi:
 *      1) primi n+m + 1 (l'uno è dovuto al fatto che all'inizio c'è il valore dell'obiettivo) valori del vettore dei costi a 0 (cudaMemsetAsync)
 *      2) ultimi m valori della prima riga a 1 (kernel)
 *      3) copia della matrice dei vincoli originale dalla prima riga di tabular->table sulle prime n colonne (cudaMemcpy2DAsync)
 *      4) riempimento delle successive m colonne con identità (kernel)
 *      5) riempimento ultime m colonne con identità (kernel)
 *      6) copia del vettore dei termini noti nel vettore degli indicatori a partire dall'indice 1 (cudaMemcpyAsync)
 *      7) riempimento vettore della base con numeri progressivi da n+m a n+2m-1 (kernel)
 */
void fillTableu(tabular_t *tabular, int *base)
{

    /**
     * Punto 1: primi n + m valori a 0 della funzione dei costi
     */

    cudaStream_t firstStream;
    HANDLE_ERROR(cudaStreamCreate(&firstStream));
    int sizeToSetZero = tabular->problem->vars + tabular->problem->constraints + 1;
    HANDLE_ERROR(cudaMemsetAsync(tabular->costsVector, 0, BYTE_SIZE(sizeToSetZero), firstStream));

    /**
     * Punto 2: ultimi m valori della prima riga a 1 (kernel)
     */

    cudaStream_t secondStream;
    HANDLE_ERROR(cudaStreamCreate(&secondStream));

    setVectorToOne<<<BL(tabular->problem->constraints), THREADS, 0, secondStream>>>(tabular->costsVector + sizeToSetZero, tabular->problem->constraints);

    /**
     * Punto 3: copia della matrice dei vincoli originale dalla seconda riga di tabular->table sulle prime n colonne (cudaMemcpy2DAsync)
     */
    cudaStream_t thirdStream;
    HANDLE_ERROR(cudaStreamCreate(&thirdStream));

    HANDLE_ERROR(cudaMemcpy2DAsync(
        tabular->constraintsMatrix,               // destinazione
        tabular->pitch,                           // pitch della destinazione
        tabular->problem->constraintsMatrix,      // fonte
        BYTE_SIZE(tabular->problem->constraints), // pitch della fonte
        BYTE_SIZE(tabular->problem->constraints), // larghezza matrice
        tabular->problem->vars,                   // altezza matrice
        cudaMemcpyDefault,                        // tipo
        thirdStream                               // stream
        ));

    /**
     * Punto 4 e 5: riempimento delle successive m colonne con identità (kernel)
     */
    cudaStream_t fourthStream;
    HANDLE_ERROR(cudaStreamCreate(&fourthStream));

    fillMatrix<<<BL(tabular->cols), THREADS, 0, fourthStream>>>(
        ROW(tabular->constraintsMatrix, tabular->problem->vars, tabular->pitch),
        tabular->cols, // colonne della matrice
        tabular->pitch // pitch
    );

    /**
     * Punto 6: copia del vettore dei termini noti nel vettore degli indicatori (cudaMemcpyAsync)
     */
    cudaStream_t fifthStream;
    HANDLE_ERROR(cudaStreamCreate(&fifthStream));
    HANDLE_ERROR(cudaMemcpyAsync(tabular->indicatorsVector,                // puntatore al vettore destinazione (vettore indicatori)
                                 tabular->problem->knownTermsVector,       // fonte
                                 BYTE_SIZE(tabular->problem->constraints), // dimensione in byte del vettore
                                 cudaMemcpyDefault,                        // tipo
                                 fifthStream));                            // stream

    /**
     * Punto 7: riempimento vettore della base con numeri progressivi da n+m a n+2m-1 (kernel)
     */
    cudaStream_t sixthStream;
    HANDLE_ERROR(cudaStreamCreate(&sixthStream));

    fillBaseVector<<<BL(tabular->cols), THREADS, 0, sixthStream>>>(
        base,
        tabular->cols,
        tabular->problem->vars + tabular->problem->constraints);

    HANDLE_KERNEL_ERROR();

    // distruggiamo gli stream
    HANDLE_ERROR(cudaStreamDestroy(firstStream));
    HANDLE_ERROR(cudaStreamDestroy(secondStream));
    HANDLE_ERROR(cudaStreamDestroy(thirdStream));
    HANDLE_ERROR(cudaStreamDestroy(fourthStream));
    HANDLE_ERROR(cudaStreamDestroy(fifthStream));
    HANDLE_ERROR(cudaStreamDestroy(sixthStream));
}

// TODO: esiste una versione migliore?
/**
 * Controlla se il problema è degenere
 * @return DEGENERATE se degenere, FEASIBLE altrimenti
 */
int checkIfDegenerate(tabular_t *tabular, int *base)
{

    int *checkDegenere_h, *checkDegenere_map;
    HANDLE_ERROR(cudaHostAlloc(&checkDegenere_h, sizeof(int), cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostGetDevicePointer(&checkDegenere_map, checkDegenere_h, 0));
    HANDLE_ERROR(cudaMemset(checkDegenere_map, 0, sizeof(int)));

    checkVector<<<BL(tabular->cols), THREADS>>>(base,
                                                tabular->cols,
                                                (tabular->problem->vars) + (tabular->cols),
                                                ((tabular->problem->vars) + 2 * (tabular->cols)),
                                                checkDegenere_map);
    HANDLE_KERNEL_ERROR();

    int checkDegenere = *checkDegenere_h;

    HANDLE_ERROR(cudaFreeHost(checkDegenere_h));

    if (checkDegenere > 0)
    {
        return DEGENERATE;
    }

    return FEASIBLE;
}

/**
 * Passaggi per la fase 1
 * Dati m il numero di vincoli e n il numero di variabili
 *   1) Riempimento tabella (su cuda stream diversi)
 *   2) Esecuzione algoritmo di gauss per portare la funzione obiettivo in funzione delle variabili non in base
 *   3) Eseguo l'algoritmo di soluzione fino all'ottimo (file a parte)
 *   4) Controllo il primo valore del vettore degli indicatori: se < 0 problema infeasibleù
 *   5) Se è presente in base un valore x tale che n+m <= x < n+2m, il problema è degenere (fare su un kernel?)
 *   6) Proseguo a fase 2
 */
int phase1(tabular_t *tabular, int *base_h, int* base_dev)
{
    /**
     * Fase 1: riempimento del tableu
     */

    fillTableu(tabular, base_dev);

#ifdef DEBUG
    //FILE *file = fopen("debugPhase1.txt", "w");

    fprintf(stdout, "\n\n\nTableu nella situazione iniziale\n");

    printTableauToStream(stdout, tabular);

    // printing della base
    fprintf(stdout, "Vettore della base\n");
    for (int i = 0; i < tabular->cols; i++)
    {
        fprintf(stdout, "%d\t", base_h[i]);
    }
#endif

    /**
     * Fase 2: eliminazione di gauss
     */

    updateObjectiveFunction(tabular, base_dev);

#ifdef DEBUG
    fprintf(stdout, "\n\n\nTableu dopo l'eliminazione di gauss\n");
    printTableauToStream(stdout, tabular);
#endif

    /**
     * Fase 3: lancio del solver
     */

    solve(tabular, base_h);

#ifdef DEBUG
    fprintf(stdout, "\n\n\nTableu dopo il lancio del primo solver\n");
    printTableauToStream(stdout, tabular);
    fprintf(stdout, "Vettore della base\n");
    for (int i = 0; i < tabular->cols; i++)
    {
        fprintf(stdout, "%d\t", base_h[i]);
    }
#endif

    /**
     * Fase 4: controllo infattibilità
     */

    TYPE firstKnownTermsValue;
    HANDLE_ERROR(cudaMemcpy(&firstKnownTermsValue, tabular->costsVector, BYTE_SIZE(1), cudaMemcpyDeviceToHost));
    if (firstKnownTermsValue < 0)
    {
        return INFEASIBLE;
    }

    /**
     * Fase 5: controllo degenere: se è presente in base un valore x tale che n+m <= x < n+2m, il problema è degenere
     */
    return checkIfDegenerate(tabular, base_dev);
}

/**
 * Passaggi per la fase 2
 * Dati m il numero di vincoli e n il numero di variabili
 *
 * 1) Riduco il numero di righe del tableu da n+2m+1 a n+m+1 (basta aggiornare il valore in tabular->rows) (perchè lavoriamo sulla trasposta)
 * 2) Uso due stream per riempire la funzione obiettivo
 *      1) Setto il primo elemento della funzione di costo a 0 (errore?)
 *      2) Copio il vettore dei coefficienti nei seguenti n elementi della prima riga (cudaMemcpyAsync)
 *      2) Imposto i restanti m elementi a 0 (cudaMemsetAsync)
 * 3) Esprimo la funzione obiettivo in termini delle variabili non di base (vedi es. istogramma)
 * 4) Eseguo l'algoritmo di soluzione fino all'ottimo (file a parte)
 */
int phase2(tabular_t *tabular, int *base_h, int *base_dev)
{
    /**
     * Fase 1: riduzione del numero di colonne
     */
    tabular->rows -= tabular->cols;

#ifdef DEBUG
    //FILE *file = fopen("debugPhase2.txt", "w");
    fprintf(stdout, "\n\n\nTableu dopo aggiornamento colonne in phase2\n");
    printTableauToStream(stdout, tabular);

    fprintf(stdout, "Vettore della base\n");
    for (int i = 0; i < tabular->cols; i++)
    {
        fprintf(stdout, "%d\t", base_h[i]);
    }
#endif

    /**
     * Fase 2: riempimento vettore costi su due stream diversi
     */

    cudaStream_t firstStream;
    cudaStream_t secondStream;

    HANDLE_ERROR(cudaStreamCreate(&firstStream));
    HANDLE_ERROR(cudaStreamCreate(&secondStream));

    // usiamo un solo stream per memsetAsync, converrebbe fare con 3?
    HANDLE_ERROR(cudaMemsetAsync((tabular->costsVector + 1) + tabular->problem->vars, 0, BYTE_SIZE(tabular->cols), firstStream)); // ultimi m elementi a 0

    HANDLE_ERROR(cudaMemcpyAsync(tabular->costsVector + 1,
                                 tabular->problem->objectiveFunction,
                                 BYTE_SIZE(tabular->problem->vars),
                                 cudaMemcpyDefault,
                                 secondStream));
    negateVector<<<BL(tabular->problem->vars), THREADS, 0, secondStream>>>(tabular->costsVector + 1, tabular->problem->vars);
    HANDLE_KERNEL_ERROR();

    HANDLE_ERROR(cudaStreamDestroy(firstStream));
    HANDLE_ERROR(cudaStreamDestroy(secondStream));

#ifdef DEBUG
    fprintf(stdout, "\n\n\nTableu dopo riempimento funzione obiettivo in phase2\n");
    printTableauToStream(stdout, tabular);
#endif

    /**
     *  Fase 3: Eliminazione di gauss per esprimere la funzione obiettivo in termini delle variabili non di base
     */

    updateObjectiveFunction(tabular, base_dev);

#ifdef DEBUG
    fprintf(stdout, "\n\n\nTableu dopo eliminazione di gauss in phase2\n");
    printTableauToStream(stdout, tabular);
#endif

/**
 * Fase 4: Esecuzione dell'algoritmo di risoluzione
 */

// in ogni caso solo un solve viene lanciato
#ifdef DEBUG
    int esito = solve(tabular, base_h);
    fprintf(stdout, "\n\n\nTableu dopo seconda esecuzione del solver\n\n\n");
    printTableauToStream(stdout, tabular);
    //fclose(file);
    return esito;
#endif

    return solve(tabular, base_h);
}

__inline__ void unregisterMemory(int *base_h, problem_t *problem)
{
    HANDLE_ERROR(cudaHostUnregister(problem->constraintsMatrix));
    HANDLE_ERROR(cudaHostUnregister(problem->knownTermsVector));
    HANDLE_ERROR(cudaHostUnregister(problem->objectiveFunction));
    HANDLE_ERROR(cudaFreeHost(base_h));
}

void getSolutionHost(tabular_t *tabular, int *base, TYPE *solution, TYPE *optimalValue)
{
    HANDLE_ERROR(cudaMemcpy(optimalValue, tabular->costsVector, BYTE_SIZE(1), cudaMemcpyDefault));

    TYPE *dev_solution;
    HANDLE_ERROR(cudaHostRegister(solution, BYTE_SIZE(tabular->problem->vars), cudaHostRegisterMapped));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_solution, solution, 0));

    HANDLE_ERROR(cudaMemset(dev_solution, 0, BYTE_SIZE(tabular->problem->vars)));
    
    getSolution<<<BL(tabular->cols), THREADS>>>(tabular->indicatorsVector, base, tabular->cols, dev_solution, tabular->problem->vars);
    HANDLE_KERNEL_ERROR();

    // qui dipende da cosa passiamo in solution, se è gia alloccata solution ci basta un memcpy, altrimenti anche un malloc, supponiamo sia già alloccata
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

    int result = phase1(tabular, base_h, base_map);
    if (result != FEASIBLE)
    {
        unregisterMemory(base_h, problem);
        return result;
    }

    result = phase2(tabular, base_h, base_map);
    if (result != FEASIBLE)
    {
        unregisterMemory(base_h, problem);
        return result;
    }

    getSolutionHost(tabular, base_map, solution, optimalValue);

    unregisterMemory(base_h, problem);
    return result;
}