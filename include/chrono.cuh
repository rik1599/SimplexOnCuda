#pragma once

#include <stdio.h>
#include "tabular.cuh"

void initCsv();
void initCsvBenchmark(int vars, int constraints);

void start(tabular_t* tabular, const char *operation);

void stop();

void closeCsv();
