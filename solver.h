#pragma once
#include "tabular.cuh"

/** Dato un tableau istanziato lo risolve (o trova una soluzione ottima o dichiara il problema UNBOUNDED)
 * 
 * 1) Test ottimalità: la prima riga del tableau è >= 0
 *      Se true -> problema FEASIBLE
 *      altrimenti prosegui
 * 2) Cerca minimo nella prima riga -> colPivot
 * 3) Controllo di unbounding: colPivot è <= 0
 *      Se true -> problema UNBOUNDED
 *      altrimenti prosegui
 * 4) Costruisco il vettore degli indicatori: divido l'ultima colonna con colPivot
 * 5) Cerco il minimo nel vettore degli indicatori -> rowPivot
 * 6) base[rowPivot] = colPivot
 * 7) pivot = A[rowPivot][colPivot] e recPivot = 1/pivot
 * 8) Copia rowPivot e colPivot in due vettori a parte
 * 9) Aggiornamento tableau per tile. Per ogni A[y][x]
 *      se y = rowPivot allora A[y][x] = A[y][x] * recPivot
 *      altrimenti  A[y][x] = - colPivot[y] * rowPivot[x] * recPivot + A[y][x]
 * 10) Ricomincia da 1
 * 
 * Per i punti 3, 4, 8 lavoro su singole colonne. Per evitare accessi strided copio la colonna in un 
 * vettore in global memory utilizzando questo metodo:
 * 1) Calcolo in quali "tile" (32x32) si trova la colonna da copiare
 * 2) Lancio una grid di blocchi grandi 32x32 (la grid è grande rows/32)
 * 3) Ogni thread copia un elemento della matrice in shared memory (matrice 32x33 per evitare bank conflict)
 * 4) I thread con indice y=0, copiano l'elemento della colonna cercata nel vettore in global memory
 */
int solve(tabular_t* tabular, int* base);