#include <cstdio>
#include <iostream>

#include "Cuda.hpp"
#include "Vec3.hpp"
#include "Renderer.hpp"

using namespace std;

__global__ void raytraceCudaKernel() {
    Vec3 v(1, 2, 3);
    
    RayTracer<CPURayTracer> raytracer;
}


void launchCudaKernel() {
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(1, 1, 1);
    
    raytraceCudaKernel<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
}


