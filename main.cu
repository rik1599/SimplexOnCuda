#include <stdio.h>
#include "problem.h"
#include "twoPhaseMethod.h"
#include "macro.h"

void setupDevice();

int main(int argc, const char *argv[])
{
    printf("Starting....\n");

    setupDevice();

    FILE *file = NULL;
    int vars;
    int constraints;
    int seed = 0;
    problem_t *problem;
    bool casuallyGenerated = false;

    if (argc < 2)
    {
        fprintf_s(stderr, "Argomenti insufficienti\n");
        exit(-1);
    }

    if (strcmp(argv[1], "-f") == 0)
    {
        printf("Leggo problema da file\n");
        if (fopen_s(&file, argv[2], "r") != 0)
        {
            fprintf(stderr, "Errore nell'apertura del file\n");
            exit(-1);
        }

        problem = readProblemFromFile(file);
        fclose(file);
    }
    else if (strcmp(argv[1], "-r") == 0)
    {
        printf("Genero problema casuale\n");
        vars = atoi(argv[2]);
        constraints = atoi(argv[3]);
        srand(time(NULL));
        seed = argc > 4 ? atoi(argv[4]) : rand();
        printf("Seed: %d\n", seed);
        casuallyGenerated = true;
        problem = generateRandomProblem(vars, constraints, seed);
    }
    else if (strcmp(argv[1], "-rf") == 0)
    {
        printf("Leggo dati generatore da file\n");
        if (fopen_s(&file, argv[2], "r") != 0)
        {
            fprintf(stderr, "Errore nell'apertura del file");
            exit(-1);
        }
        problem = readRandomProblemFromFile(file);
        fclose(file);
    }
#ifdef DEBUG
    printProblemToStream(stdout, problem);
#endif

    TYPE *solution = (TYPE *)(malloc(BYTE_SIZE(problem->vars)));
    TYPE optimalValue = 0;
    FILE *fileSolution = NULL;
    if (fopen_s(&fileSolution, "solution.txt", "w") != 0)
    {
        fprintf(stderr, "Errore nell'apertura del file\n");
        exit(-1);
    }

    printf("Resolving....\n");
    switch (twoPhaseMethod(problem, solution, &optimalValue))
    {
    case INFEASIBLE:
        printf("Problem INFEASIBLE!\n");
        break;

    case UNBOUNDED:
        printf("Problem UNBOUNDED!\n");
        break;

    case DEGENERATE:
        printf("Problem DEGENERATE!\n");
        break;

    default:
        printf("\nProblem solved!\n");

        for (size_t i = 0; i < problem->vars; i++)
        {
            fprintf_s(fileSolution, "%lf\n", solution[i]);
        }
        fprintf_s(fileSolution, "\nOptimal value: %lf\n", optimalValue);

        fclose(fileSolution);

        // se il problema era casuale ci salviamo i dati per la generazione
        if (casuallyGenerated)
        {
            // salviamo il file con nome l'ora attuale... pensare a qualcosa di meglio
            FILE *saveFile;
            fopen_s(&saveFile, "randomProblemData.txt", "w");
            fprintf_s(saveFile, "%d %d %d", vars, constraints, seed);
            fclose(saveFile);
        }
        break;
    }

    free(solution);
    freeProblem(problem);
}

void setupDevice()
{
    if (TYPE_SIZE == 8)
    {
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        printf("Bank size set to 8 byte\n");
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.canMapHostMemory)
    {
        fprintf_s(stderr, "Device cannot map memory!\n");
        exit(-1);
    }
    cudaSetDeviceFlags(cudaDeviceMapHost);
}
