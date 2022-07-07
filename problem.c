#include "problem.h"

#include <stdlib.h>

problem_t* readProblemFromFile(FILE* file)
{
    problem_t* problem = (problem_t*)malloc(sizeof(problem_t));
    int nVars = 0;
    int nConstraints = 0;

    //Leggo il numero di variabili e vincoli del problema dalla prima riga del file
	fscanf_s(file, "%d %d", &nVars, &nConstraints);
    problem->constraints = nConstraints;
    problem->vars = nVars;

    //Leggo il vettore dei costi dalla seconda riga del file
    problem->objectiveFunction = (TYPE*)malloc(BYTE_SIZE(nVars));
    for (size_t i = 0; i < nVars; i++)
    {
        fscanf_s(file, "%lf", &problem->objectiveFunction[i]);
    }

    //Leggo la matrice delle costanti e il vettore dei termini noti
    problem->constraintsMatrix = (TYPE*)malloc(BYTE_SIZE(nVars * nConstraints));
    problem->knownTermsVector = (TYPE*)malloc(BYTE_SIZE(nConstraints));
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