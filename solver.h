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
 * 8) Copia rowPivot e colPivot in due vettori a parte (Per copiare una riga si usa cudaMemcpy)
 * 9) Aggiornamento tableau per tile. Per ogni A[y][x]
 *      se y = rowPivot allora A[y][x] = A[y][x] * recPivot
 *      altrimenti  A[y][x] = - colPivot[y] * rowPivot[x] * recPivot + A[y][x]
 * 10) Ricomincia da 1
 * 
 * SI PREVEDA UN MODO PER MISURARE IL TEMPO NECESSARIO A SVOLGERE UN GIRO
 * E RIPORTARE I DATI SU UN CSV PER L'ANALISI DEI TEMPI
 */
int solve(tabular_t* tabular, int* base);