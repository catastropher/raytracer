#include <cstdio>
#include <iostream>
#include <vector>

#include "Cuda.hpp"
#include "Vec3.hpp"
#include "Renderer.hpp"

using namespace std;

#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 4
#define THREADS_IN_BLOCK BLOCK_WIDTH * BLOCK_HEIGHT

struct GPURayTracer {
    Scene* scene;
    
    CUDA_CALLABLE GPURayTracer(Scene* scene_) : scene(scene_) { }
    
    CUDA_CALLABLE Intersection<Triangle> findClosestIntersectedTriangle(const Ray& ray, const Shape* lastReflection) {
        Intersection<Triangle> closestIntersection;
        Intersection<Triangle> triIntersection;
        
        for(Triangle* tri = scene->triangles.begin(); tri != scene->triangles.end(); ++tri) {
            if(tri != lastReflection && tri->calculateRayIntersections(ray, &triIntersection) > 0) {
                //if(triIntersection.distanceFromRayStartSquared < closestIntersection.distanceFromRayStartSquared)
                //    closestIntersection = triIntersection;
                
                
                closestIntersection = minimum(triIntersection, closestIntersection);
            }
        }
        
        return closestIntersection;
    }
    
    CUDA_CALLABLE Intersection<Sphere> findClosestIntersectedSphere(const Ray& ray, const Shape* lastReflection) {
        Intersection<Sphere> closestIntersection;
        Intersection<Sphere> sphereIntersections[2];
        
        //printf("Total triangles: %d\n", (int)scene->triangles.total);
        
        for(Sphere* sphere = scene->spheres.begin(); sphere != scene->spheres.end(); ++sphere) {
            if(sphere == lastReflection)
                continue;
            
            int count = sphere->calculateRayIntersections(ray, sphereIntersections);
            
            if(count == 1) {
                closestIntersection = minimum(closestIntersection, sphereIntersections[0]);
            }
            else if(count == 2) {
                closestIntersection = minimum(closestIntersection, minimum(sphereIntersections[0], sphereIntersections[1]));
            }
        }
        
        return closestIntersection;
    }
    
    CUDA_CALLABLE Intersection<Shape> findClosestIntersectedShape(const Ray& ray, const Shape* lastReflection) {
        return minimum(
            findClosestIntersectedTriangle(ray, lastReflection).toGenericShapeIntersection(),
            findClosestIntersectedSphere(ray, lastReflection).toGenericShapeIntersection()
        );
    }
};

typedef RayTracer<GPURayTracer> Tracer;

std::vector<void*> cudaAllocatedMemory;

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
    
    printf("Transfered size: %d (%d)\n", (int)size, (int)sizeof(T));
    
    attemptCudaMalloc(&deviceList.list, size);
    cudaMemcpy(deviceList.list, hostList.list, size, cudaMemcpyHostToDevice);
    
    deviceList.total = hostList.total;
    
    return deviceList;
}

CUDATriangleList convertTriangleGeometryListToCUDATriangleList(GeometryList<Triangle>& triangles) {
    CUDATriangleList deviceList;
    
    GeometryList<float> triangleGeometry(triangles.total * cudaTriangleListSize());
    GeometryList<Material> materials(triangles.total);
    GeometryList<Color> colors(triangles.total);
    
    for(int i = 0; i < triangles.total; ++i) {
        Triangle& tri = triangles.list[i];
        float* triangleStart = triangleGeometry.list + i * cudaTriangleListSize();
        
        for(int j = 0; j < 3; ++j) {
            cudaTriangleListVX(triangleStart, j) = tri.p[i].x;
            cudaTriangleListVY(triangleStart, j) = tri.p[i].y;
            cudaTriangleListVZ(triangleStart, j) = tri.p[i].z;
        }
        
        cudaTriangleListPlaneA(triangleStart) = tri.plane.normal.x;
        cudaTriangleListPlaneB(triangleStart) = tri.plane.normal.y;
        cudaTriangleListPlaneC(triangleStart) = tri.plane.normal.z;
        cudaTriangleListPlaneD(triangleStart) = tri.plane.d;
        
        materials.list[i] = triangles.list[i].material;
        colors.list[i] = triangles.list[i].color;
    }
    
    deviceList.triangleGeometry = copyGeometryListToGPU(triangleGeometry);
    deviceList.triangleMaterials = copyGeometryListToGPU(materials);
    deviceList.triangleColors = copyGeometryListToGPU(colors);
    
    triangleGeometry.cleanup();
    materials.cleanup();
    colors.cleanup();
    
    return deviceList;
}

Scene createSceneOnDevice(Scene hostScene) {
    Scene deviceScene = hostScene;
    
    deviceScene.triangles = GeometryList<Triangle>(); //copyGeometryListToGPU(hostScene.triangles);
    deviceScene.spheres = copyGeometryListToGPU(hostScene.spheres);
    deviceScene.lights = copyGeometryListToGPU(hostScene.lights);
    
    deviceScene.cudaTriangles = convertTriangleGeometryListToCUDATriangleList(hostScene.triangles);
    
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
    dim3 gridSize(w / BLOCK_WIDTH, h / BLOCK_HEIGHT, 1);
    dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    
    cudaDeviceReset();
    
    Color* deviceFrameBuffer;
    Tracer* deviceTracer = createRayTracerOnDevice(angle, w, h, scene, deviceFrameBuffer);
    
    printf("Triangles on CPU: %d\n", scene.triangles.total);
    raytraceCudaKernel<<<gridSize, blockSize>>>(deviceTracer);
    cudaDeviceSynchronize();
    
    copyFrameBufferFromDeviceToHost(deviceFrameBuffer, hostRenderer);
    
    cudaMemoryCleanup();
}


