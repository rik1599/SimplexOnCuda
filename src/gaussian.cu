#include "gaussian.cuh"
#include "error.cuh"

#define THREADS 512
#define BL(N) min((N + THREADS - 1) / THREADS, 1024)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val == 0.0)
        return __longlong_as_double(old);
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/** Implementa in parallelo operazioni del tipo
 * mat[0] = mat[0] - (\sum_{i=i}^{rows-1} coefficient[i]*mat[i])
 *
 * In pratica sottrae al vettore a riga 0 del tableau (valori della funzione obiettivo)
 * TUTTE le altre righe, ognuna opportunatamente moltiplicata per un coefficiente.
 *
 * L'operazione viene fatta "per tile"
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
            TYPE value = idX < cols && idX + blockDim.x < cols ? *(INDEX(mat, idY, idX, pitch)) * coefficients[idX] +
                                                                     *(INDEX(mat, idY, idX + blockDim.x, pitch)) * coefficients[idX + blockDim.x]
                                                               : (idX < cols ? *(INDEX(mat, idY, idX, pitch)) * coefficients[idX]
                                                                             : 0);

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
#else
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
__global__ void gaussianElimination(TYPE *mat, TYPE *objectiveFunction, TYPE *coefficients, int rows, int cols, size_t pitch)
{
    for (int idX = threadIdx.x + (blockIdx.x * blockDim.x * 2); idX < cols; idX += 2 * gridDim.x * blockDim.x)
    {
        for (int idY = threadIdx.y + blockIdx.y * blockDim.y; idY < rows; idY += gridDim.y * blockDim.y)
        {
            atomicAdd(&objectiveFunction[idY], -(idX + blockDim.x < cols ? ((*(INDEX(mat, idY, idX, pitch)) * coefficients[idX]) +
                                                                            (*(INDEX(mat, idY, (idX + blockDim.x), pitch))) * coefficients[idX + blockDim.x])
                                                                         : (*(INDEX(mat, idY, idX, pitch))) * coefficients[idX]));
        }
    }
}
#endif

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
    for (int idX = threadIdx.x + blockIdx.x * blockDim.x;
         idX < baseSize;
         idX += gridDim.x * blockDim.x)
    {
        coefficients[idX] = firstRow[base[idX]];
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
    dim3 grid = dim3(
        1,
        (tabular->rows + (block.y - 1)) / block.y);

    gaussianElimination<<<grid, block>>>(
        tabular->table,
        tabular->costsVector,
        coefficientVector,
        tabular->rows,
        tabular->cols,
        tabular->pitch);
    HANDLE_KERNEL_ERROR();

    HANDLE_ERROR(cudaFree(coefficientVector));
}