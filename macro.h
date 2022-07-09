#define DEBUG

#define TYPE double

#define TYPE_SIZE sizeof(TYPE)
#define BYTE_SIZE(count) count * TYPE_SIZE

#define TILE_DIM 32
#define GRID_DIM(size) size < TILE_DIM ? 1 : size/TILE_DIM
#define TAB_ELEMENT(vet, row, col, pitch) (TYPE*) ((char*) vet + row * pitch) + col