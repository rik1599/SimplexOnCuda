#include <stdio.h>
#include "problem.h"

#define DEBUG

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

    freeProblem(problem);
}