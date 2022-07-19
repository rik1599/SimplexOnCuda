#pragma once

#include "macro.h"

/** Kernel che genera casualmente una matrice di tipo TYPE
 * 
 * @param outMatrix - puntatore alla matrice da generare (allocata con cudaMallocPitch)
 * @param rows - numero di righe della matrice
 * @param cols - numero di colonne della matrice
 */
__global__ void generateMatrix(TYPE* outMatrix, int rows, int cols);

/** Kernel che genera casualmente un vettore di tipo TYPE
 * 
 * @param outVet - puntatore al vettore da generare
 * @param size - dimensione del vettore
 */
__global__ void generateVector(TYPE* outVet, int size);