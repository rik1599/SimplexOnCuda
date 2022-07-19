#include "tabular.cuh"
#include "macro.h"

#define INFEASIBLE -1
#define UNBOUNDED -2
#define DEGENERATE -3
#define FEASIBLE 0

/** Esegue il metodo del simplesso a due fasi
 * per risolvere il problema dato in input
 * 
 * @param problem problem_t* - puntatore al problema da risolvere
 * @param solution TYPE* - puntatore a un vettore nel quale verranno poi riportati i valori ottimi delle variabili
 * @param optimalValue TYPE* - puntatore a una variabile nel quale verrà riportato il valore ottimo della funzione obiettivo
 * 
 * @return -1 se il problema non ha soluzione, -2 se è unbounded, -3 se il problema è degenere, 0 se ha una soluzione
 */
int twoPhaseMethod(problem_t* problem, TYPE* solution, TYPE* optimalValue);
