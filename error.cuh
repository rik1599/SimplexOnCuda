#pragma once

void HandleError(cudaError_t err, const char *file, int line);
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

void checkKernelError(const char *file, int line);
#define HANDLE_KERNEL_ERROR() (checkKernelError(__FILE__, __LINE__))
