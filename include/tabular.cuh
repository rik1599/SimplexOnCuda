#pragma once

#include "problem.h"

typedef struct
{
    // Riferimento al problema iniziale
    problem_t* problem;

    // Tableau memorizzato in global memory (per colonne)
    TYPE* table;

    //Puntatore al vettore degli indicatori in table (vettore-colonna dei termini noti)
    TYPE* indicatorsVector;

    //Puntatore alla matrice dei vincoli (dalla riga 1 in poi di table)
    TYPE* constraintsMatrix;

    // Larghezza "reale" di table (in byte) in global memory.
    size_t pitch;

    //Numero di righe di table
    int rows;

    //Numero di colonne di table
    int cols;

    //Vettore dei costi (coefficienti della funzione obiettivo)
    TYPE* costsVector;

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
