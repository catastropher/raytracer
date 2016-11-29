#include <cstdio>

#include "Cuda.hpp"

__global__ void raytraceCudaKernel() {
    printf("Hello from CUDA kernal!");
}


void launchCudaKernel() {
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(1, 1, 1);
    
    raytraceCudaKernel<<<gridSize, blockSize>>>();
}


