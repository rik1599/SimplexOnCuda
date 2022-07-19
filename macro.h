#define DEBUG

#define TYPE double

#define TYPE_SIZE sizeof(TYPE)
#define BYTE_SIZE(count) count * TYPE_SIZE

#define ROW(vet, row, pitch) (TYPE*) ((char*) vet + row * pitch)
#define INDEX(vet, row, col, pitch) ROW(vet, row, pitch) + col