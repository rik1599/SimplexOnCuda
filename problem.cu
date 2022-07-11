#include "problem.h"

#include <stdlib.h>

problem_t* mallocProblem(int nVars, int nConstraints)
{
    problem_t* problem = (problem_t*)malloc(sizeof(problem_t));
    
    problem->constraints = nConstraints;
    problem->vars = nVars;
    
    problem->objectiveFunction = (TYPE*)malloc(BYTE_SIZE(nVars));
    problem->constraintsMatrix = (TYPE*)malloc(BYTE_SIZE(nVars * nConstraints));
    problem->knownTermsVector = (TYPE*)malloc(BYTE_SIZE(nConstraints));

    return problem;
}

problem_t* readProblemFromFile(FILE* file)
{
    int nVars = 0;
    int nConstraints = 0;

    //Leggo il numero di variabili e vincoli del problema dalla prima riga del file
	fscanf_s(file, "%d %d", &nVars, &nConstraints);

    problem_t* problem = mallocProblem(nVars, nConstraints);

    //Leggo il vettore dei costi dalla seconda riga del file
    for (size_t i = 0; i < nVars; i++)
    {
        fscanf_s(file, "%lf", &problem->objectiveFunction[i]);
    }

    //Leggo la matrice delle costanti e il vettore dei termini noti
    for (size_t i = 0; i < nConstraints; i++)
    {
        for (size_t j = 0; j < nVars; j++)
        {
            fscanf_s(file, "%lf", &problem->constraintsMatrix[i * nVars + j]);
        }
        fscanf_s(file, "%lf\n", &problem->knownTermsVector[i]);
    }

    return problem;
}

problem_t* generateRandomProblem(int width, int height)
{
    problem_t* problem = mallocProblem(width, height);

    /*L'idea è quella di generare i tre vettori utilizzando 
    tre stream e poi trasferire la matrice creata in memoria
    (per una questione di uniformità del codice e della procedura di test)
    */

    return problem;
}

void printProblemToStream(FILE* Stream, problem_t* problem)
{
    int nVars = problem->vars;
    int nConstraints = problem->constraints;

    fprintf(Stream, "max ");
    for (size_t i = 0; i < nVars - 1; i++)
    {
        fprintf(Stream, "%.2lf * x%zd + ", problem->objectiveFunction[i], i + 1);
    }
    fprintf(Stream, "%.2lf * x%d\n", problem->objectiveFunction[nVars - 1], nVars);
    fprintf(Stream, "subject to \n");

    for (size_t i = 0; i < nConstraints; i++)
    {
        for (size_t j = 0; j < nVars; j++)
        {
            fprintf(Stream, "%.2lf * x%zd + ", problem->constraintsMatrix[i * nVars + j], j + 1);
        }
        fprintf(Stream, "%.2lf * x%d ", problem->constraintsMatrix[i * nVars + (nVars-1)], nVars);
		fprintf(Stream, "<= %.2lf\n", problem->knownTermsVector[i]);
    }
    
}

void freeProblem(problem_t* problem)
{
    free(problem->constraintsMatrix);
    free(problem->knownTermsVector);
    free(problem->objectiveFunction);
}