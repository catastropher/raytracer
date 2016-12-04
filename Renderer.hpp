#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdio>

#include "Material.hpp"
#include "Light.hpp"
#include "Ray.hpp"
#include "Shape.hpp"
#include "Triangle.hpp"
#include "Sphere.hpp"
#include "Scene.hpp"
#include "config.hpp"

#ifdef __WITH_SDL__
  #include "quickcg.h"
#endif

static inline float degToRadians(float deg) {
    const float PI = 3.1415926535897;
    return deg * PI / 180.0;
}

template<typename T>
CUDA_CALLABLE T minimum(const T& a, const T& b) {
    return (a < b ? a : b);
}

// #ifdef __WITH_CUDA__
// template <> CUDA_CALLABLE float minimum<float>(const float& a, const float& b) {
//     return fminf(a, b);
// }
// #endif

struct CPURayTracer {
    Scene* scene;
    
    CUDA_CALLABLE CPURayTracer(Scene* scene_) : scene(scene_) { }
    
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

struct Renderer {
    Color* frameBuffer;
    
    Color backgroundColor;
    
    int screenW, screenH;
    float viewAngle;
    float distToScreen;
    
    void initialize(float angle, int w, int h) {
#ifdef __WITH_SDL__
        QuickCG::screen(w, h);
#endif
        screenW = w;
        screenH = h;
        
        frameBuffer = new Color[w * h];
        
        viewAngle = angle;
        distToScreen = (w / 2 ) / tan(degToRadians(angle / 2));
        
        //std::cout << "Dist to screen: " << distToScreen << std::endl;
    }
    
#ifdef __WITH_CUDA__
    void initializeCuda(float angle, int w, int h);
#endif
    
    void saveFrameBuffer(std::string fileName) {
        FILE* file = fopen(fileName.c_str(), "wb");
        
        if(!file)
            throw "Failed to open " + fileName + " for writing";
        
        fprintf(file, "%d %d\n", screenW, screenH);
        
        for(int i = 0; i < screenW * screenH; ++i) {
            Color c = frameBuffer[i];
            fprintf(file, "%f %f %f\n", c.x, c.y, c.z);
        }
        
        fclose(file);
    }
    
    void loadFrameBuffer(std::string fileName) {
        FILE* file = fopen(fileName.c_str(), "rb");
        
        if(!file)
            throw "Failed to open " + fileName + " for reading";
        
        fscanf(file, "%d %d", &screenW, &screenH);
        
        initialize(viewAngle, screenW, screenH);
        
        for(int i = 0; i < screenW * screenH; ++i) {
            Color& c = frameBuffer[i];
            fscanf(file, "%f %f %f", &c.x, &c.y, &c.z);
        }
        
        fclose(file);
    }
    
    void displayFrameBuffer() {
#ifdef __WITH_SDL__
        for(int i = 0; i < screenW * screenH; ++i) {
            Color c = frameBuffer[i];
            QuickCG::pset(i % screenW, i / screenW, QuickCG::ColorRGB(c.x, c.y, c.z));
        }
#endif
    }
};


template<typename Tracer>
struct RayTracer {    
    Scene scene;
    
    Renderer renderer;
    Tracer tracer;
    
    CUDA_CALLABLE RayTracer() : tracer(&scene) { }
    
    CUDA_DEVICE Ray calculateRayForPixelOnScreen(int x, int y) {
        Vec3 pixelPos(x - renderer.screenW / 2, y - renderer.screenH / 2, renderer.distToScreen);
        
        return Ray(scene.camPosition, pixelPos + scene.camPosition);
    }
    
    CUDA_DEVICE Color calculateOnlySpecularHightlights(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = Vec3(0, 0, 0);
        
        if(shape != NULL) {
            for(Light* light = scene.lights.begin(); light != scene.lights.end(); ++light) {
                result = result + light->calculateSpecular(shape->material, pointOnObj, scene.camPosition, objNormal);
            }
        }
        
        return result;
    }
    
    CUDA_DEVICE Color calculateLighting(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Shape tempShape;
        
        if(shape == NULL)
            shape = &tempShape;
        
        Color result = shape->color * scene.ambientLightIntensity;
        
        for(Light* light = scene.lights.begin(); light != scene.lights.end(); ++light) {
            Ray lightRay(pointOnObj, light->pos);
            Intersection<Shape> closestIntersection;
            
            //if(!findNearestRayIntersection(lightRay, shape, closestIntersection))
                result = result + light->evaluatePhongReflectionModel(shape->material, shape->color, objNormal, pointOnObj, scene.camPosition);
        }
        
        return result.maxValue(1.0);
    }
    
    CUDA_DEVICE bool findNearestRayIntersection(const Ray& ray, const Shape* lastReflection, Intersection<Shape>& closestIntersection) {        
        closestIntersection = tracer.findClosestIntersectedShape(ray, lastReflection).toGenericShapeIntersection();
        return closestIntersection.shape != NULL;
    }
    
#if defined(__WITH_CUDA__) && __CUDACC__
    CUDA_DEVICE Color traceRay(const Ray& ray, int recursionDepth, const Shape* lastReflection) {
        Intersection<Shape> closestIntersection;
        bool hitAtLeastOneObject = findNearestRayIntersection(ray, lastReflection, closestIntersection);
        
        __shared__ bool needReflection[THREADS_IN_BLOCK];
        
        int id = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
        
        needReflection[id] = closestIntersection.shape != NULL && closestIntersection.shape->material.reflective;
        
        __syncthreads();
        
        bool reflect = false;
        
        for(int i = 0; i < THREADS_IN_BLOCK; ++i)
            reflect |= needReflection[i];
        
        __syncthreads();
       
        Color lighting = calculateLighting(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos);
        
        if(reflect) {
            Color reflectedRayColor = calculateReflectedRayColor(ray, closestIntersection, recursionDepth);
            
            if(closestIntersection.shape != NULL) {
                 if(closestIntersection.shape->material.reflective)
                     return reflectedRayColor;
                 else
                     return lighting;
            }
            else {
                return renderer.backgroundColor;
            }
        }
        
        if(!hitAtLeastOneObject)
            return renderer.backgroundColor;
        
        return lighting;
    }
#else
    CUDA_DEVICE Color traceRay(const Ray& ray, int recursionDepth, const Shape* lastReflection) {        
        Intersection<Shape> closestIntersection;
        bool hitAtLeastOneObject = findNearestRayIntersection(ray, lastReflection, closestIntersection);
        
        if(!hitAtLeastOneObject)
            return renderer.backgroundColor;
        
        if(closestIntersection.shape->material.reflective)
            return calculateReflectedRayColor(ray, closestIntersection, recursionDepth);
        
        Color lighting = calculateLighting(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos);
        
        return lighting;
    }
#endif
    
    CUDA_DEVICE Color calculateReflectedRayColor(const Ray& ray, Intersection<Shape>& closestIntersection, int recursionDepth) {
        if(recursionDepth >= 1)
            return renderer.backgroundColor;
        
        Ray reflectedRay = ray.reflectAboutNormal(closestIntersection.normal, closestIntersection.pos);
                    
        Color reflectedRayColor = traceRay(reflectedRay, recursionDepth + 1, closestIntersection.shape);
        
        float dot = -ray.dir.dot(closestIntersection.normal);
        float alpha = .9;
        float fresnelEffect = pow(1 - dot, 3) * (1 - alpha) + 1 * alpha;
        
        return (reflectedRayColor * fresnelEffect + calculateOnlySpecularHightlights(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos)).maxValue(1.0);
    }
    
    CUDA_DEVICE void raytraceSingleRay(int x, int y) {
        Ray ray = calculateRayForPixelOnScreen(x, y);
                
        Color rayColor = traceRay(ray, 0, NULL) * 255;
        renderer.frameBuffer[y * renderer.screenW + x] = rayColor;
    }
    
    CUDA_DEVICE void raytrace() {
        for(int i = 0; i < renderer.screenH; ++i) {
            for(int j = 0; j < renderer.screenW; ++j) {
                raytraceSingleRay(j, i);
                
                Color rayColor = renderer.frameBuffer[i * renderer.screenW + j];
                
#ifdef __WITH_SDL__
                QuickCG::pset(j, i, QuickCG::ColorRGB(rayColor.x, rayColor.y, rayColor.z));
#endif
            }
           
#ifdef __WITH_SDL__
            QuickCG::redraw();
            QuickCG::redraw();
#endif
            //std::cout << (i * 100 / renderer.screenH) << "%" << std::endl;
        }
    }
};

