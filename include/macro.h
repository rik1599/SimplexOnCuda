#pragma once

#define DEBUG

#define TYPE double

#define TYPE_SIZE sizeof(TYPE)
#define BYTE_SIZE(count) count * TYPE_SIZE

__host__ __device__ __inline__ TYPE* ROW(TYPE* vet, int row, size_t pitch)
{
    return (TYPE*) ((char*) vet + row * pitch);
}

__host__ __device__ __inline__ TYPE* INDEX(TYPE* vet, int row, int col, size_t pitch)
{
    return ROW(vet, row, pitch) + col;
}