#include "error.cuh"
#include <stdio.h>
#include <stdlib.h>

void HandleError(cudaError_t err, const char * file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

void checkKernelError(const char * file, int line)
{
	cudaDeviceSynchronize();
	HandleError(cudaGetLastError(), file, line);
}
