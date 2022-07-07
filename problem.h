#pragma once

#include <stdio.h>

#define TYPE double
#define TYPE_SIZE sizeof(TYPE)
#define BYTE_SIZE(count) count * TYPE_SIZE

/** Struttura per memorizzare i problemi letti da file (o generati in automatico).
 *  Si assumono che i problemi siano tutti nella forma
 *  max cx s.t. Ax <= b
*/
typedef struct problem
{
    //Matrice delle costanti (linearizzata per righe)
	TYPE* constraintsMatrix;

    //Vettore dei termini noti
	TYPE* knownTermsVector;

    //Vettore con i coefficienti della funzione obiettivo
	TYPE* objectiveFunction;

	//Numero di variabili del problema (larghezza della matrice)
	int vars;

	//Numero di vincoli del problemi (altezza della matrice)
	int constraints;
} problem_t;

/** Legge il problema da un file di numeri separati da spazi.
 * La prima riga contiene rispettivamente il numero di variabili e il numero di costanti
 * La seconda riga contiene i coefficienti della funzione obiettivo
 * Ogni riga dalla terza in poi contiene una riga della matrice dei vincoli.
 * L'ultimo elemento di ogni riga è il corrispettivo elemento del vettore dei termini noti
 * 
 * @param file FILE* - puntatore al file da cui leggere i dati
 * @return problem_t* - puntatore al problema
*/
problem_t* readProblemFromFile(FILE* file);

/** Genera casualmente un problema dati il numero di variabili e di vincoli
 * 
 * @param nVars int - numero di variabili del problema
 * @param nConstraints int - numero di vincoli del problema
 * @return problem_t* - puntatore al problema
 */
problem_t* generateRandomProblem(int nVars, int nConstraints);

/** Stampa il problema su uno stream nella forma
 * max c0 x1 + ... + cn-1 xn
 * subject to
 * a00 x1 + ... + a0n <= b0
 * ...
 * an0 x1 + ... + ann <= bn
 * 
 * @param Stream FILE* - puntatore al file o allo stream su cui stampare
 * @param problem problem_t* - puntatore al problema da stampare
 */
void printProblemToStream(FILE* Stream, problem_t* problem);

/** Esegue il free della memoria del problema
 * 
 * @param problem problem_t* - puntatore al problema
 */
void freeProblem(problem_t* problem);