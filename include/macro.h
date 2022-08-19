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

/** Funzione per comparare due numeri in virgola mobile
 *  @param x
 *  @param y se non specificato, si compara x con 0
 *  @param epsilon precisione desiderata. Di default 1e-9
 * 
 *  @return 0 se x == y (considerata la precisione), -1 se x < y, 1 se x > 1
 */
__host__ __device__ __inline__ int compare(double x, double y = 0.0, double epsilon = 1e-9)
{
    if (abs(x - y) < epsilon)
    {
        return 0;
    }
    else if (x < y)
    {
        return -1;
    }
    else
    {
        return 1;
    }
    
}