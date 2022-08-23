#pragma once

#include "macro.h"
#include "error.cuh"

/**
 * Genera un array casuale di dimensione size attraverso un kernel cuda nella locazione di memoria device passata in input
 * La generazione avviene in maniera asincrona sullo stream ritornato dalla funzione
 *
 * @param dst - puntatore alla locazione in memoria device dove salvare il vettore
 * @param size - la dimensione dell'array da generare
 * @param seed - seed della generazione casuale
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum - limite superiore alla generazione degli elementi
 * @return - lo stream su cui viene eseguita la generazione
 */
cudaStream_t *generateVectorInParallelAsync(TYPE *dst, int size, unsigned int, double minimum, double maximum);

/**
 * Genera una matrice casuale attraverso un kernel cuda in una locazione di memoria specificata, la matrice è linearizzata per colonne
 * L'allocazione in device memory è bloccante, ma asyncrona dopo, conviene chiamarla prima di altre chiamate asincrone
 *
 * @param dst - puntatore alla locazione di memoria dove salvare la matrice (la matrice è linearizzata per colonne)
 * @param width - la dimensione orizzontale della matrice da generare
 * @param height - la dimensione verticale della matrice da generare
 * @param seed - seed per la generazione casuale per riprodurre i risulatati
 * @param minimum - limite inferiore alla generazione degli elementi
 * @param maximum -limite superiore alla generazione degli elementi
 * @return - lo stream su cui viene eseguita la generazione
 */
cudaStream_t *generateMatrixInParallelAsync(TYPE *dst, int width, int height, unsigned int, double minimum, double maximum);