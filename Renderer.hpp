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

#ifdef __WITH_SDL__
  #include "quickcg.h"
#endif

static inline float degToRadians(float deg) {
    const float PI = 3.1415926535897;
    return deg * PI / 180.0;
}

struct CPURayTracer {
    Scene* scene;
    
    CUDA_CALLABLE CPURayTracer(Scene* scene_) : scene(scene_) { }
    
    Intersection<Triangle> findClosestIntersectedTriangle(const Ray& ray, const Shape* lastReflection) {
        Intersection<Triangle> closestIntersection;
        Intersection<Triangle> triIntersection;
        
        for(Triangle* tri = scene->triangles.begin(); tri != scene->triangles.end(); ++tri) {
            if(tri != lastReflection && tri->calculateRayIntersections(ray, &triIntersection) != 0) {
                closestIntersection = std::min(closestIntersection, triIntersection);
            }
        }
        
        return closestIntersection;
    }
    
    Intersection<Sphere> findClosestIntersectedSphere(const Ray& ray, const Shape* lastReflection) {
        Intersection<Sphere> closestIntersection;
        Intersection<Sphere> sphereIntersections[2];
        
        for(Sphere* sphere = scene->spheres.begin(); sphere != scene->spheres.end(); ++sphere) {
            if(sphere == lastReflection)
                continue;
            
            int count = sphere->calculateRayIntersections(ray, sphereIntersections);
            
            if(count == 1) {
                closestIntersection = std::min(closestIntersection, sphereIntersections[0]);
            }
            else if(count == 2) {
                closestIntersection = std::min(closestIntersection, std::min(sphereIntersections[0], sphereIntersections[1]));
            }
        }
        
        return closestIntersection;
    }
    
    Intersection<Shape> findClosestIntersectedShape(const Ray& ray, const Shape* lastReflection) {
        return std::min(
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
    
    CUDA_CALLABLE Ray calculateRayForPixelOnScreen(int x, int y) {
        Vec3 pixelPos(x - renderer.screenW / 2, y - renderer.screenH / 2, renderer.distToScreen);
        
        return Ray(scene.camPosition, pixelPos + scene.camPosition);
    }
    
    CUDA_CALLABLE Color calculateOnlySpecularHightlights(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = Vec3(0, 0, 0);
        
        for(Light* light = scene.lights.begin(); light != scene.lights.end(); ++light) {
            result = result + light->calculateSpecular(shape->material, pointOnObj, scene.camPosition, objNormal);
        }
        
        return result;
    }
    
    CUDA_CALLABLE Color calculateLighting(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = shape->color * scene.ambientLightIntensity;
        
        for(Light* light = scene.lights.begin(); light != scene.lights.end(); ++light) {
            Ray lightRay(pointOnObj, light->pos);
            Intersection<Shape> closestIntersection;
            
            if(!findNearestRayIntersection(lightRay, shape, closestIntersection))
                result = result + light->evaluatePhongReflectionModel(shape->material, shape->color, objNormal, pointOnObj, scene.camPosition);
        }
        
        return result.maxValue(1.0);
    }
    
    CUDA_CALLABLE bool findNearestRayIntersection(const Ray& ray, const Shape* lastReflection, Intersection<Shape>& closestIntersection) {        
        closestIntersection = tracer.findClosestIntersectedShape(ray, lastReflection).toGenericShapeIntersection();
        return closestIntersection.shape != NULL;
    }
    
    CUDA_CALLABLE Color traceRay(const Ray& ray, int recursionDepth, const Shape* lastReflection) {
        Intersection<Shape> closestIntersection;
        bool hitAtLeastOneObject = findNearestRayIntersection(ray, lastReflection, closestIntersection);
        
        if(!hitAtLeastOneObject)
            return renderer.backgroundColor;
        
        if(closestIntersection.shape->material.reflective)
            return calculateReflectedRayColor(ray, closestIntersection, recursionDepth);
        
        return calculateLighting(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos);
    }
    
    CUDA_CALLABLE Color calculateReflectedRayColor(const Ray& ray, Intersection<Shape>& closestIntersection, int recursionDepth) {
        if(recursionDepth > 5)
            return renderer.backgroundColor;
        
        Ray reflectedRay = ray.reflectAboutNormal(closestIntersection.normal, closestIntersection.pos);
                    
        Color reflectedRayColor = traceRay(reflectedRay, recursionDepth + 1, closestIntersection.shape);
        
        float dot = -ray.dir.dot(closestIntersection.normal);
        float alpha = .9;
        float fresnelEffect = pow(1 - dot, 3) * (1 - alpha) + 1 * alpha;
        
        return (reflectedRayColor * fresnelEffect + calculateOnlySpecularHightlights(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos)).maxValue(1.0);
    }
    
    void raytrace() {
        for(int i = 0; i < renderer.screenH; ++i) {
            for(int j = 0; j < renderer.screenW; ++j) {
                Ray ray = calculateRayForPixelOnScreen(j, i);
                
                Color rayColor = traceRay(ray, 0, NULL) * 255;
                renderer.frameBuffer[i * renderer.screenW + j] = rayColor;
                
#ifdef __WITH_SDL__
                QuickCG::pset(j, i, QuickCG::ColorRGB(rayColor.x, rayColor.y, rayColor.z));
#endif
            }
           
#ifdef __WITH_SDL__
            QuickCG::redraw();
            QuickCG::redraw();
#endif
            std::cout << (i * 100 / renderer.screenH) << "%" << std::endl;
        }
    }
};

