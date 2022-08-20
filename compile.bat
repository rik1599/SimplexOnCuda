nvcc -rdc=true -o debug/SimplexOnCuda -Iinclude/ -arch=sm_60^
 main.cu ^
 src/error.cu ^
 src/problem.cu ^
 src/tabular.cu ^
 src/solver.cu ^
 src/twoPhaseMethod.cu ^
 src/generator.cu ^
 src/reduction.cu ^
 src/gaussian.cu