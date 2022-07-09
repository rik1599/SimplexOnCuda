#pragma once

#include "problem.h"

typedef struct tabular
{
    // Riferimento al problema iniziale
    problem_t* problem;

    // Matrice dei vincoli memorizzata in global memory
    TYPE* table;

    /** Larghezza "reale" di table (in byte) in global memory.
     */
    size_t pitch;

    //Numero di righe di table
    int rows;

    //Numero di colonne di table
    int cols;

    //Ultima colonna di table (vettore dei termini noti)
    TYPE* indicatorCol;

} tabular_t;

/** Istanzia il problema in forma tabellare.
 * 
 * @param problem problem_t* - puntatore al problema da tabellare
 * @return istanza di tabular_t* da passare al solver
 */
tabular_t* newTabular(problem_t* problem);

/** Stampa il problema in forma tabellare su uno Stream
 * 
 * @param Stream FILE* - puntatore al file o lo stream su cui stampare
 * @param tabular tabular_t* - puntatore al problema in forma tabellare da stampare
 */
void printTableauToStream(FILE* Stream, tabular_t* tabular);

/** Esegue il free della memoria per il tabular
 * 
 * @param tabular tabular_t* - puntatore al problema in forma tabellare
 */
void freeTabular(tabular_t* tabular);
