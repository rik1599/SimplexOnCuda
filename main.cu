#include <stdio.h>
#include "problem.h"
#include "solver.h"
#include "macro.h"

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        fprintf_s(stderr, "Argomenti mancanti!");
        exit(-1);
    }

    FILE* file;
    if (fopen_s(&file, argv[1], "r") != 0)
    {
        fprintf(stderr, "Errore nell'apertura del file");
		exit(-1);
    }

    problem_t* problem = readProblemFromFile(file);
    fclose(file);

    #ifdef DEBUG
    printProblemToStream(stdout, problem);
    #endif

    setupDevice();
    
    TYPE* solution = (TYPE*)(malloc(BYTE_SIZE(problem->vars)));
    TYPE optimalValue = 0;

    switch (solve(problem, solution, &optimalValue))
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
        printf("Problem solved!\nVariables values: ");
        for (size_t i = 0; i < problem->vars; i++)
        {
            printf("%lf\t", solution[i]);
        }
        printf("\nOptimal value: %lf\n", optimalValue);
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
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.canMapHostMemory)
    {
        exit(-1);
    }
    cudaSetDeviceFlags(cudaDeviceMapHost);
}