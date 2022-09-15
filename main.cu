#include <stdio.h>
#include <time.h>
#include "problem.h"
#include "twoPhaseMethod.h"
#include "macro.h"

void setupDevice();
problem_t *randomInput(int vars, int contraints, int seed);
void saveRandomInput(int vars, int constraints, int seed);

int main(int argc, const char *argv[])
{
    printf("Starting...\n");

    setupDevice();

    if (argc < 2)
    {
        fprintf(stderr, "Not enough arguments!\n");
        exit(-1);
    }

    problem_t *problem = NULL;
    if (strcmp(argv[1], "-f") == 0)
    {
        printf("Reading problem from file...\n");
        FILE *file = openFile(argv[2], "r");
        problem = readProblemFromFile(file);
        fclose(file);
    }
    else if (strcmp(argv[1], "-r") == 0)
    {
        problem = randomInput(atoi(argv[2]), atoi(argv[3]), argc > 4 ? atoi(argv[4]) : time(NULL));
    }
    else if (strcmp(argv[1], "-rs") == 0)
    {
        int seed = argc > 4 ? atoi(argv[4]) : time(NULL);
        problem = randomInput(atoi(argv[2]), atoi(argv[3]), seed);
        saveRandomInput(atoi(argv[2]), atoi(argv[3]), seed);
    }
    else if (strcmp(argv[1], "-rf") == 0)
    {
        printf("Reading seed from file\n");
        FILE *file = openFile(argv[2], "r");
        problem = readRandomProblemFromFile(file);
        fclose(file);
    }else if (strcmp(argv[1], "-t") == 0)
    {
        #ifdef TIMER
            enableBenchmarkMode();
        #endif
        fprintf(stderr, "Running a benchmark (max 8192*8192)... \n\n\n");
        int constraints = 256;
        time_t start =  time(NULL);

        while(constraints <= 8192){
            int vars = 256;
            while(vars <= 8192){
                fprintf(stdout, "\nCurrent matrix: %d*%d\n\n", vars, constraints);
                int seed = vars*100+constraints + (vars == 1024 && constraints == 8192 ? 1 : 0);
                problem_t *benchmarkProblem = generateRandomProblem(vars, constraints, seed, +1, +100);
                TYPE *solution = (TYPE *)(malloc(BYTE_SIZE(benchmarkProblem->vars)));
                TYPE optimalValue = 0;
                twoPhaseMethod(benchmarkProblem, solution, &optimalValue);
                freeProblem(benchmarkProblem);
                free(solution);
                vars *= 2;
            }
            constraints *= 2;
        }
        time_t end =  time(NULL);
        fprintf(stdout, "Benchmark terminato...\n Sono stati necessari %.3lfs", (double) end-start);
        return 0;
    }
#ifdef DEBUG
    printProblemToStream(stdout, problem);
#endif

    TYPE *solution = (TYPE *)(malloc(BYTE_SIZE(problem->vars)));
    TYPE optimalValue = 0;
    FILE *fileSolution = openFile("..\\data\\solution.txt", "w");

    printf("Resolving....\n");
    switch (twoPhaseMethod(problem, solution, &optimalValue))
    {
    case INFEASIBLE:
        printf("\nProblem INFEASIBLE!\n");
        break;

    case UNBOUNDED:
        printf("\nProblem UNBOUNDED!\n");
        break;

    case DEGENERATE:
        printf("\nProblem DEGENERATE!\n");
        break;

    default:
        printf("\nProblem solved!\n");

        for (size_t i = 0; i < problem->vars; i++)
        {
            fprintf(fileSolution, "%lf\n", solution[i]);
        }
        fprintf(fileSolution, "\nOptimal value: %lf\n", optimalValue);

        fclose(fileSolution);
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
        fprintf(stderr, "Device cannot map memory!\n");
        exit(-1);
    }
    cudaSetDeviceFlags(cudaDeviceMapHost);
}

problem_t *randomInput(int vars, int constraints, int seed)
{
    printf("Generating random problem with %d variables, %d contraints with seed: %d\n", vars, constraints, seed);
    return generateRandomProblem(vars, constraints, seed, -100, +100);
}

void saveRandomInput(int vars, int constraints, int seed)
{
    time_t timer = time(NULL);
    char time_str[20];
    struct tm *tm_info = localtime(&timer);
    strftime(time_str, 20, "%Y%m%d%H%M", tm_info);

    char fileName[50];
    sprintf(fileName, "..\\data\\examples\\random_%s.txt", time_str);
    FILE *saveFile = openFile(fileName, "w");
    fprintf(saveFile, "%d %d %d", vars, constraints, seed);
    fclose(saveFile);
}