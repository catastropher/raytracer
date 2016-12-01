#include <cstdio>
#include <iostream>

#include "Cuda.hpp"
#include "Vec3.hpp"
#include "Renderer.hpp"

using namespace std;

typedef RayTracer<CPURayTracer> Tracer;

void Renderer::initializeCuda(float angle, int w, int h) {
    screenW = w;
    screenH = h;
    
    cudaMalloc(&frameBuffer, sizeof(Color) * w * h);
    
    viewAngle = angle;
    distToScreen = (w / 2 ) / tan(degToRadians(angle / 2));
}

__global__ void raytraceCudaKernel(Tracer* tracer) {
    printf("Tracer dim: %d %d\n", tracer->renderer.screenW, tracer->renderer.screenH);
    
    printf("Total triangles: %d\n", tracer->scene.triangles.total);
}

template<typename T>
GeometryList<T> copyGeometryListToGPU(GeometryList<T> hostList) {
    GeometryList<T> deviceList;
    
    size_t size = hostList.end() - hostList.begin();
    cudaMalloc(&deviceList.list, size);
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

Tracer* createRayTracerOnDevice(float angle, int screenW, int screenH, Scene scene) {
    Renderer deviceRenderer;
    deviceRenderer.initializeCuda(angle, screenW, screenH);
    
    Scene deviceScene = createSceneOnDevice(scene);
    
    Tracer hostTracer;
    hostTracer.renderer = deviceRenderer;
    hostTracer.scene = deviceScene;
    
    Tracer* deviceTracer;
    cudaMalloc(&deviceTracer, sizeof(Tracer));
    cudaMemcpy(deviceTracer, &hostTracer, sizeof(Tracer), cudaMemcpyHostToDevice);
    
    return deviceTracer;
}

void launchCudaKernel(float angle, int w, int h, Scene scene) {
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(1, 1, 1);
    
    Tracer* deviceTracer = createRayTracerOnDevice(angle, w, h, scene);
    
    printf("Triangles on CPU: %d", scene.triangles.total);
    raytraceCudaKernel<<<gridSize, blockSize>>>(deviceTracer);
    cudaDeviceSynchronize();
}


