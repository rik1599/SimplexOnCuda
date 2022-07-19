nvcc -rdc=true -o debug/SimplexOnCuda -Iinclude/ \
 main.cu \
 src/error.cu \
 src/problem.cu \
 src/tabular.cu \
 src/solver.cu \
 src/twoPhaseMethod.cu \
 src/generator.cu \
 src/reduction.cu