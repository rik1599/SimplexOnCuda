#include <stdio.h>
#include "problem.h"
#include "twoPhaseMethod.h"
#include "macro.h"
#include "time.h"

void setupDevice();

int main(int argc, const char *argv[])
{
    printf("Starting....\n");

    setupDevice();

    FILE *file = NULL;
    int vars;
    int constraints;
    problem_t *problem;

    if (argc == 2)
    {
        printf("Leggo problema da file\n");
        if (fopen_s(&file, argv[1], "r") != 0)
        {
            fprintf(stderr, "Errore nell'apertura del file");
            exit(-1);
        }

        problem = readProblemFromFile(file);
        fclose(file);
    }
    else if (argc == 3)
    {
        printf("Genero problema casuale\n");
        vars = atoi(argv[1]);
        constraints = atoi(argv[2]);

        problem = generateRandomProblem(vars, constraints, 0);
    }
    else
    {
        fprintf_s(stderr, "Argomenti mancanti!");
        exit(-1);
    }

#ifdef DEBUG
    printProblemToStream(stdout, problem);
#endif

    TYPE *solution = (TYPE *)(malloc(BYTE_SIZE(problem->vars)));
    TYPE optimalValue = 0;
    FILE *fileSolution = NULL;
    if (fopen_s(&file, "solution.txt", "w") != 0)
    {
        fprintf(stderr, "Errore nell'apertura del file");
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
        printf("Problem solved!\n");

        for (size_t i = 0; i < problem->vars; i++)
        {
            fprintf_s(fileSolution, "%lf\n", solution[i]);
        }
        fprintf_s(fileSolution, "\nOptimal value: %lf\n", optimalValue);
        break;
    }
    fclose(fileSolution);
    
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
