#include "chrono.cuh"
#include "error.cuh"

static FILE* fileChrono;
static cudaEvent_t startEvent;
static cudaEvent_t stopEvent;

void initCsv()
{
    time_t timer = time(NULL);
    char time_str[20];
    struct tm *tm_info = localtime(&timer);
    strftime(time_str, 20, "%Y%m%d%H%M%S.%d", tm_info);

    char fileName[50];
    sprintf(fileName, "..\\data\\measures\\times_%s.txt", time_str);
    fileChrono = openFile(fileName, "w");
    fprintf(fileChrono, "vars,contraints,operation,elapsed_time\n");

    HANDLE_ERROR(cudaEventCreate(&startEvent));
    HANDLE_ERROR(cudaEventCreate(&stopEvent));
}

void initCsvBenchmark(int vars, int constraints)
{
    char fileName[50];
    sprintf(fileName, "..\\data\\measures\\benchmark_%d_%d.txt", vars, constraints);
    fileChrono = openFile(fileName, "w");
    fprintf(fileChrono, "vars,contraints,operation,elapsed_time\n");

    HANDLE_ERROR(cudaEventCreate(&startEvent));
    HANDLE_ERROR(cudaEventCreate(&stopEvent));
}

void start(tabular_t* tabular, const char *operation)
{
    fprintf(fileChrono, "%d,%d,%s,", tabular->rows, tabular->cols, operation);
    cudaEventRecord(startEvent, 0);
}

void stop()
{
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    fprintf(fileChrono, "%f\n", elapsedTime * 1000);
}

void closeCsv()
{
    fclose(fileChrono);
    HANDLE_ERROR(cudaEventDestroy(startEvent));
    HANDLE_ERROR(cudaEventDestroy(stopEvent));
}