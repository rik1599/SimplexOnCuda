#pragma once

#include <stdio.h>

#define TYPE double
#define TYPE_SIZE sizeof(TYPE)
#define BYTE_SIZE(count) count * TYPE_SIZE

/* Struttura per memorizzare i problemi letti da file.
*  Si assumono che i problemi siano tutti nella forma
*  max cx s.t. Ax <= b
*/
typedef struct problem
{
    //Matrice delle costanti (linearizzata per righe)
	TYPE* constraintsMatrix;

    //Vettore dei termini noti
	TYPE* knownTermsVector;

    //Vettore 
	TYPE* objectiveFunction;
	int vars;
	int constraints;
} problem_t;

problem_t* readProblemFromFile(FILE* file);

void printProblemToStream(FILE* Stream, problem_t* problem);

void freeProblem(problem_t* problem);