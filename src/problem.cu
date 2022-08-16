#include "problem.h"
#include "error.cuh"
#include "generator.cuh"

#include <stdlib.h>

#define MINIMUM_GENERATION 0
#define MAXIMUM_GENERATION +1000


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
            fscanf_s(file, "%lf", &problem->constraintsMatrix[j * nConstraints + i]);
        }
        fscanf_s(file, "%lf\n", &problem->knownTermsVector[i]);
    }

    return problem;
}

problem_t* generateRandomProblem(int width, int height, int seed)
{
    
    /*L'idea è quella di generare i tre vettori utilizzando 
    tre stream e poi trasferire la matrice creata in memoria
    (per una questione di uniformità del codice e della procedura di test)
    */

    //allocazione problema in memoria
    problem_t* problem = mallocProblem(width, height);                           
    
    //dal seed iniziale generiamo tre seed di partenza per i kernel generatori.
    srand(seed);
    int seedOne = rand();
    int seedTwo = rand() + problem->constraints*problem->vars;
    int seedThree = rand() + problem->constraints*problem->vars + problem->constraints;

    #ifdef DEBUG
        printf("Attualmente i seed sono: %d, %d, %d\n", seedOne, seedTwo, seedThree);
    #endif

    //per la gestione asincrona è necessario settare la memoria host in page locked
    HANDLE_ERROR(cudaHostRegister(problem->constraintsMatrix, BYTE_SIZE(problem->vars * problem->constraints), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->objectiveFunction, BYTE_SIZE(problem->vars), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->knownTermsVector, BYTE_SIZE(problem->constraints), cudaHostRegisterDefault));

    /*
    * Generazione della matrice, se si volesse tornare alla linearizzazione per colonne => invertire costraints e vars
    */
    cudaStream_t* streamMatrice = generateMatrixInParallelAsync(problem->constraintsMatrix,
                                problem->constraints, //dato che la vogliamo linearizzata per colonne generiamo una trasposta
                                problem->vars,
                                seedThree,
                                MINIMUM_GENERATION,
                                MAXIMUM_GENERATION);

    //Generazione casuale termini noti
    cudaStream_t* streamKnownTermsVector = 
                generateVectorInParallelAsync(problem->knownTermsVector,
                                            problem->constraints,
                                            (unsigned long long) seedOne,
                                            MINIMUM_GENERATION,
                                            MAXIMUM_GENERATION);

    //Generazione casuale funzione obiettivo
    cudaStream_t* streamObjectiveFunction =
                generateVectorInParallelAsync(problem->objectiveFunction,
                                                problem->vars,
                                                (unsigned long long) seedTwo,
                                                MINIMUM_GENERATION,
                                                MAXIMUM_GENERATION);

    //sincronizziamo tutti gli stream
    cudaDeviceSynchronize();

    HANDLE_ERROR(cudaStreamDestroy(*streamMatrice));
    HANDLE_ERROR(cudaStreamDestroy(*streamKnownTermsVector));
    HANDLE_ERROR(cudaStreamDestroy(*streamObjectiveFunction));

    //rendiamo la memoria non più registered
    HANDLE_ERROR(cudaHostUnregister(problem->constraintsMatrix));
    HANDLE_ERROR(cudaHostUnregister(problem->objectiveFunction));
    HANDLE_ERROR(cudaHostUnregister(problem->knownTermsVector));
    
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
        for (size_t j = 0; j < nVars - 1; j++)
        {
            fprintf(Stream, "%.2lf * x%zd + ", problem->constraintsMatrix[j * nConstraints + i], j + 1);
        }
        fprintf(Stream, "%.2lf * x%d ", problem->constraintsMatrix[(nVars - 1) * nConstraints + i], nVars);
		fprintf(Stream, "<= %.2lf\n", problem->knownTermsVector[i]);
    }
}

void freeProblem(problem_t* problem)
{
    free(problem->constraintsMatrix);
    free(problem->knownTermsVector);
    free(problem->objectiveFunction);
}

