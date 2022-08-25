#include "problem.h"
#include "error.cuh"
#include "generator.cuh"

#include <stdlib.h>

problem_t *mallocProblem(int nVars, int nConstraints)
{
    problem_t *problem = (problem_t *)malloc(sizeof(problem_t));

    problem->constraints = nConstraints;
    problem->vars = nVars;

    problem->objectiveFunction = (TYPE *)malloc(BYTE_SIZE(nVars));
    problem->constraintsMatrix = (TYPE *)malloc(BYTE_SIZE(nVars * nConstraints));
    problem->knownTermsVector = (TYPE *)malloc(BYTE_SIZE(nConstraints));
    return problem;
}

problem_t *readProblemFromFile(FILE *file)
{
    int nVars = 0;
    int nConstraints = 0;

    // Leggo il numero di variabili e vincoli del problema dalla prima riga del file
    fscanf(file, "%d %d", &nVars, &nConstraints);

    problem_t *problem = mallocProblem(nVars, nConstraints);

    // Leggo il vettore dei costi dalla seconda riga del file
    for (size_t i = 0; i < nVars; i++)
    {
        fscanf(file, "%lf", &problem->objectiveFunction[i]);
    }

    // Leggo la matrice delle costanti e il vettore dei termini noti
    for (size_t i = 0; i < nConstraints; i++)
    {
        for (size_t j = 0; j < nVars; j++)
        {
            fscanf(file, "%lf", &problem->constraintsMatrix[j * nConstraints + i]);
        }
        fscanf(file, "%lf\n", &problem->knownTermsVector[i]);
    }

    return problem;
}

problem_t *generateRandomProblem(int nVars, int nConstraints, unsigned int seed, int minGenerator, int maxGenerator)
{

    /**
     * L'idea è quella di generare i tre vettori utilizzando
     * tre stream e poi trasferire la matrice creata in memoria
     * (per una questione di uniformità del codice e della procedura di test)
     */

    // allocazione problema in memoria
    problem_t *problem = mallocProblem(nVars, nConstraints);

    // dal seed iniziale generiamo tre seed di partenza per i kernel generatori: ogni generatore avrà
    // un proprio seed generato casualmente a partire da quello di partenza, facendo in questo modo abbiamo la ripetibilità.
    srand(seed);

    unsigned int seedOne = rand();
    unsigned int seedTwo = rand();
    unsigned int seedThree = rand();

#ifdef DEBUG
    printf("Seed used: %d, %d, %d\n", seedOne, seedTwo, seedThree);
#endif

    // per la gestione asincrona è necessario settare la memoria host in page locked
    HANDLE_ERROR(cudaHostRegister(problem->constraintsMatrix, BYTE_SIZE(problem->vars * problem->constraints), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->objectiveFunction, BYTE_SIZE(problem->vars), cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(problem->knownTermsVector, BYTE_SIZE(problem->constraints), cudaHostRegisterDefault));

    // utilizziamo la memoria mapped per i due vettori
    TYPE *objectiveFunction_map;
    TYPE *knownTermsVector_map;

    HANDLE_ERROR(cudaHostGetDevicePointer(&objectiveFunction_map, problem->objectiveFunction, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&knownTermsVector_map, problem->knownTermsVector, 0));

    /*
     * Generazione della matrice, se si volesse tornare alla linearizzazione per righe => invertire costraints e vars
     */
    cudaStream_t *streamMatrice = generateMatrixInParallelAsync(
        problem->constraintsMatrix,
        problem->constraints, // dato che la vogliamo linearizzata per colonne generiamo una trasposta
        problem->vars,
        seedThree,
        minGenerator,
        maxGenerator);

    // Generazione casuale termini noti
    cudaStream_t *streamKnownTermsVector = generateVectorInParallelAsync(
        knownTermsVector_map,
        problem->constraints,
        (unsigned long long)seedOne,
        minGenerator,
        maxGenerator);

    // Generazione casuale funzione obiettivo
    cudaStream_t *streamObjectiveFunction = generateVectorInParallelAsync(
        objectiveFunction_map,
        problem->vars,
        (unsigned long long)seedTwo,
        minGenerator,
        maxGenerator);

    // sincronizziamo tutti gli stream
    HANDLE_KERNEL_ERROR();

    // distruggiamo gli stream
    HANDLE_ERROR(cudaStreamDestroy(*streamMatrice));
    HANDLE_ERROR(cudaStreamDestroy(*streamKnownTermsVector));
    HANDLE_ERROR(cudaStreamDestroy(*streamObjectiveFunction));

    // rendiamo la memoria non più registered
    HANDLE_ERROR(cudaHostUnregister(problem->constraintsMatrix));
    HANDLE_ERROR(cudaHostUnregister(problem->objectiveFunction));
    HANDLE_ERROR(cudaHostUnregister(problem->knownTermsVector));

    return problem;
}

problem_t *readRandomProblemFromFile(FILE *file)
{
    int nVars = 0;
    int nConstraints = 0;
    unsigned int seed = 0;

    fscanf(file, "%d %d %u", &nVars, &nConstraints, &seed);

    return generateRandomProblem(nVars, nConstraints, seed);
}

void printProblemToStream(FILE *Stream, problem_t *problem)
{
    int nVars = problem->vars;
    int nConstraints = problem->constraints;

    fprintf(Stream, "max ");
    for (size_t i = 0; i < nVars; i++)
    {
        double value = problem->objectiveFunction[i];
        if (value >= 0)
        {
            fprintf(Stream, "+ ");
        }
        else
        {
            fprintf(Stream, "- ");
        }
        fprintf(Stream, "%.2lf X%zd ", abs(value), i + 1);
    }

    fprintf(Stream, "\nsubject to \n");

    for (size_t i = 0; i < nConstraints; i++)
    {
        for (size_t j = 0; j < nVars; j++)
        {
            double value = problem->constraintsMatrix[j * nConstraints + i];
            if (value >= 0)
            {
                fprintf(Stream, "+ ");
            }
            else
            {
                fprintf(Stream, "- ");
            }
            fprintf(Stream, "%.2lf X%zd ", abs(value), j + 1);
        }

        fprintf(Stream, "<= %.2lf\n", problem->knownTermsVector[i]);
    }
}

void freeProblem(problem_t *problem)
{
    free(problem->constraintsMatrix);
    free(problem->knownTermsVector);
    free(problem->objectiveFunction);
}
