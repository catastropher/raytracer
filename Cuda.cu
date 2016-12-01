#include <cstdio>
#include <iostream>
#include <vector>

#include "Cuda.hpp"
#include "Vec3.hpp"
#include "Renderer.hpp"

using namespace std;

typedef RayTracer<CPURayTracer> Tracer;

std::vector<void*> cudaAllocatedMemory;

#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 4

void cudaMemoryCleanup() {
    for(int i = 0; i < cudaAllocatedMemory.size(); ++i) {
        printf("Freed %llx\n", (long long)cudaAllocatedMemory[i]);
        cudaFree(cudaAllocatedMemory[i]);
    }
    
    cudaDeviceReset();
}

template<typename T>
void attemptCudaMalloc(T dest, size_t size) {
    if(cudaMalloc(dest, size) == cudaErrorMemoryAllocation) {
        cudaMemoryCleanup();
        throw "Failed to allocate memory on device";
    }
    
    cudaAllocatedMemory.push_back((void *)*dest);
}

void Renderer::initializeCuda(float angle, int w, int h) {
    screenW = w;
    screenH = h;
    
    attemptCudaMalloc(&frameBuffer, sizeof(Color) * w * h);
    
    viewAngle = angle;
    distToScreen = (w / 2 ) / tan(degToRadians(angle / 2));
}

__global__ void raytraceCudaKernel(Tracer* tracer) {
    //printf("Tracer dim: %d %d\n", tracer->renderer.screenW, tracer->renderer.screenH);
    
    //printf("Total triangles: %d\n", tracer->scene.triangles.total);
    
    //tracer->raytrace();
    tracer->raytraceSingleRay(blockIdx.x, blockIdx.y);
}

template<typename T>
GeometryList<T> copyGeometryListToGPU(GeometryList<T> hostList) {
    GeometryList<T> deviceList;
    
    size_t size = sizeof(T) * hostList.total;
    
    printf("Transfered size: %d (%d)", (int)size, sizeof(T));
    
    attemptCudaMalloc(&deviceList.list, size);
    cudaMemcpy(deviceList.list, hostList.list, size, cudaMemcpyHostToDevice);
    
    deviceList.total = hostList.total;
    
    return deviceList;
}

Scene createSceneOnDevice(Scene hostScene) {
    Scene deviceScene = hostScene;
    
    deviceScene.triangles = copyGeometryListToGPU(hostScene.triangles);
    deviceScene.spheres = copyGeometryListToGPU(hostScene.spheres);
    deviceScene.lights = copyGeometryListToGPU(hostScene.lights);
    
    return deviceScene;
}

Tracer* createRayTracerOnDevice(float angle, int screenW, int screenH, Scene scene, Color*& deviceFrameBuffer) {
    printf("Setting up renderer\n");
    Renderer deviceRenderer;
    deviceRenderer.initializeCuda(angle, screenW, screenH);
    
    printf("Setting up scene\n");
    Scene deviceScene = createSceneOnDevice(scene);
    
    Tracer hostTracer;
    hostTracer.renderer = deviceRenderer;
    hostTracer.scene = deviceScene;
    
    printf("Setting up raytracer\n");
    Tracer* deviceTracer;
    attemptCudaMalloc(&deviceTracer, sizeof(Tracer));
    
    hostTracer.tracer.scene = &deviceTracer->scene;
    
    cudaMemcpy(deviceTracer, &hostTracer, sizeof(Tracer), cudaMemcpyHostToDevice);
    
    deviceFrameBuffer = deviceRenderer.frameBuffer;
    
    printf("Done setting up device raytracer\n");
    
    return deviceTracer;
}

void copyFrameBufferFromDeviceToHost(Color* deviceFrameBuffer, Renderer& hostRenderer) {
    size_t size = sizeof(Color) * hostRenderer.screenW * hostRenderer.screenH;
    cudaMemcpy(hostRenderer.frameBuffer, deviceFrameBuffer, size, cudaMemcpyDeviceToHost);
}

void launchCudaKernel(float angle, int w, int h, Scene scene, Renderer& hostRenderer) {
    dim3 gridSize(w, h, 1);
    dim3 blockSize(1, 1, 1);
    
    cudaDeviceReset();
    
    Color* deviceFrameBuffer;
    Tracer* deviceTracer = createRayTracerOnDevice(angle, w, h, scene, deviceFrameBuffer);
    
    printf("Triangles on CPU: %d\n", scene.triangles.total);
    raytraceCudaKernel<<<gridSize, blockSize>>>(deviceTracer);
    cudaDeviceSynchronize();
    
    copyFrameBufferFromDeviceToHost(deviceFrameBuffer, hostRenderer);
    
    cudaMemoryCleanup();
}


