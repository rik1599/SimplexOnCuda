#include "solver.h"
#include "twoPhaseMethod.h"
#include "reduction.cuh"

/** Per i punti 3, 4, 8 lavoro su singole colonne. Per evitare accessi strided copio la colonna in un 
 * vettore in shared memory utilizzando questo metodo:
 * 1) Calcolo in quali "tile" (32x32) si trova la colonna da copiare
 * 2) Lancio una grid di blocchi grandi 32x32 (la grid è grande rows/32)
 * 3) Ogni thread copia un elemento della matrice in shared memory (matrice 32x33 per evitare bank conflict)
 * 4) I thread con indice y=0, copiano l'elemento della colonna cercata nel vettore in shared memory
 */
__global__ void copyColumn(TYPE* matrix, int rows, int cols, int colToCpy, volatile TYPE* sVet)
{

}

/** Costruisce il vettore degli indicatori lastCol/colPivot */
__global__ void createIndicatorVector(TYPE* mat, TYPE* lastCol, int rows, int cols, int colPivotIndex)
{

}

/** 9) Aggiornamento tableau per tile. Per ogni A[y][x]
 *      se y == rowPivotIndex allora A[y][x] = A[y][x] * recPivot
 *      altrimenti  A[y][x] = - colPivot[y] * rowPivot[x] * recPivot + A[y][x]
 */
__global__ void update(TYPE* mat, TYPE* lastCol, TYPE recPivot, int rows, int cols, int rowPivotIndex, int colPivotIndex)
{

}

int solve(tabular_t* tabular, int* base)
{
    return FEASIBLE;
}