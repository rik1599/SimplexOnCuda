#pragma once

#include "macro.h"
#include "error.cuh"

#define LINEAR_BLOCK 1024
#define LINEAR_GRID_MAX 256
#define TILE_X 32 //deve essere multiplo di 4: (byte*4 = 32 max coalescenza)
#define TILE_Y 8
#define GRID_2D_MAX_X 16
#define GRID_2D_MAX_Y 16

/** Kernel che genera casualmente una matrice di tipo TYPE
 * 
 * @param outMatrix - puntatore alla matrice da generare (allocata con cudaMallocPitch)
 * @param rows - numero di righe della matrice
 * @param cols - numero di colonne della matrice
 * @param pitch - pitch della matrice
 * @param seed - seed per la generazione casuale della matrice per riprodurre i risulatati
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum - limite superiore alla generazione degli elementi
 */
__global__ void generateMatrix(TYPE* outMatrix, int rows, int cols, size_t pitch, unsigned long long seed, double minimum, double maximum);

/** Kernel che genera casualmente un vettore di tipo TYPE
 * 
 * @param outVet - puntatore al vettore da generare
 * @param size - dimensione del vettore
 * @param seed - seed per la generazione casuale del vettore per riprodurre i risulatati
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum -limite superiore alla generazione degli elementi
 */
__global__ void generateVector(TYPE* outVet, int size, unsigned long long seed, double minimum, double maximum);

/**
 * Genera un array casuale di dimensione size attraverso un kernel cuda e lo salva in una locazione di memoria specificata
 * La generazione non è asincrona
 * 
 * @param dst - la destinazione del vettore generato casualmente
 * @param size - la dimensione dell'array da generare
 * @param seed - seed per la generazione casuale per riprodurre i risulatati
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum - limite superiore alla generazione degli elementi
 */
void generateVectorInParallel(TYPE* dst, int size, unsigned long long seed,  double minimum, double maximum);

/**
 * Genera un array casuale di dimensione size attraverso un kernel cuda e lo salva in una locazione di memoria specificata
 * La generazione avviene in maniera asincrona sullo stream ritornato dalla funzione
 * 
 * @param dst - la destinazione del vettore generato casualmente
 * @param size - la dimensione dell'array da generare
 * @param seed - seed per la generazione casuale per riprodurre i risultati
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum - limite superiore alla generazione degli elementi
 * @return - lo stream su cui viene eseguita la generazione
 */
cudaStream_t* generateVectorInParallelAsync(TYPE* dst, int size, unsigned long long seed,  double minimum, double maximum);


/**
 * Lancia il kernel per la generazione casuale di una vettore nella locazione di memoria device specificata
 * E' possibile specificare uno stream per permettere l'esecuzione asincrona
 * Al termine di tutti i task la locazione di memoria specificata in memoria device conterrà un vettore casuale
 * 
 * @param dev_ptre - la destinazione del vettore generato casualmente in memoria device
 * @param size - la dimensione dell'array da generare
 * @param seed - seed per la generazione casuale per riprodurre i risulatati
 * @param stream - puntatore allo stream sul quale lanciare il kernel, se NULL utilizza quello default
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum -limite superiore alla generazione degli elementi
 */
void runVectorGenerationKernel(TYPE* dev_ptr, int size, unsigned long long seed, cudaStream_t* stream,  double minimum, double maximum);


/**
 * Genera una matrice casuale di dimensione size attraverso un kernel cuda in una locazione di memoria specificata
 * La generazione non è asincrona ed avviene sullo stream 0
 * 
 * @param dst - puntatore alla locazione di memoria dove salvare la matrice generata (la matrice è linearizzata per colonne)
 * @param width - la dimensione orizzontale della matrice da generare
 * @param height - la dimensione verticale della matrice da generare
 * @param seed - seed per la generazione casuale per riprodurre i risulatati
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum -limite superiore alla generazione degli elementi
 */
void generateMatrixInParallel(TYPE* dst, int width, int height, unsigned long long seed,  double minimum, double maximum);

/**
 * Genera una matrice casuale di dimensione size attraverso un kernel cuda in una locazione di memoria specificata, la matrice è linearizzata per colonne
 * L'allocazione in device memory è bloccante, ma asyncrona dopo, conviene chiamarla prima di altre chiamate asincrone 
 * 
 * @param dst - puntatore alla locazione di memoria dove salvare la matrice generata (la matrice è linearizzata per colonne)
 * @param width - la dimensione orizzontale della matrice da generare
 * @param height - la dimensione verticale della matrice da generare
 * @param seed - seed per la generazione casuale per riprodurre i risulatati
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum -limite superiore alla generazione degli elementi
 * @return - lo stream su cui viene eseguita la generazione
 */
cudaStream_t* generateMatrixInParallelAsync(TYPE* dst, int width, int height, unsigned long long seed,  double minimum, double maximum);

/**
 * Lancia il kernel per la generazione casuale di una matrice nella locazione di memoria device specificata
 * E' possibile specificare uno stream per permettere l'esecuzione asincrona
 * Al termine di tutti i task la locazione di memoria specificata in memoria device conterrà una matrice casuale
 * 
 * @param dev_ptr - puntatore alla locazione in memoria device dove generare la matrice
 * @param width - la dimensione orizzontale della matrice da generare (anche se linearizzata per colonne si intende il numero di colonne)
 * @param height - la dimensione verticale della matrice da generare (anche se linearizzata per colonne si intende il numero di righe)
 * @param pitch - il pitch della matrice
 * @param seed - seed per la generazione casuale per riprodurre i risulatati
 * @param stream - puntatore allo stream sul quale lanciare il kernel, se NULL utilizza quello default
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum -limite superiore alla generazione degli elementi
 */
void runMatrixGenerationKernel(TYPE* dev_ptr, int width, int height, size_t pitch, unsigned long long seed, cudaStream_t* stream,  double minimum, double maximum);
