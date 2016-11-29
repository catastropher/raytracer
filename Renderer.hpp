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

struct CPURayTracer {
    Triangle* findClosestIntersectedTriangle();
};

static inline float degToRadians(float deg) {
    const float PI = 3.1415926535897;
    return deg * PI / 180.0;
}

struct Renderer {
    Color* frameBuffer;
    
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
        
        std::cout << "Dist to screen: " << distToScreen << std::endl;
    }
    
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

struct RayTracer {
    std::vector<Shape* > shapesInScene;
    
    Scene scene;
    
    Renderer renderer;
    
    Ray calculateRayForPixelOnScreen(int x, int y) {
        Vec3 pixelPos(x - renderer.screenW / 2, y - renderer.screenH / 2, renderer.distToScreen);
        
        return Ray(scene.camPosition, pixelPos + scene.camPosition);
    }
    
    Color calculateOnlySpecularHightlights(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = Vec3(0, 0, 0);
        
        for(Light* light = scene.lights.begin(); light != scene.lights.end(); ++light) {
            result = result + light->calculateSpecular(shape->material, pointOnObj, scene.camPosition, objNormal);
        }
        
        return result;
    }
    
    Color calculateLighting(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = shape->color * scene.ambientLightIntensity;
        
        for(Light* light = scene.lights.begin(); light != scene.lights.end(); ++light) {
            Ray lightRay(pointOnObj, light->pos);
            Intersection closestIntersection;
            
            if(!findNearestRayIntersection(lightRay, shape, closestIntersection))
                result = result + light->evaluatePhongReflectionModel(shape->material, shape->color, objNormal, pointOnObj, scene.camPosition);
        }
        
        return result.maxValue(1.0);
    }
    
    bool findNearestRayIntersection(const Ray& ray, const Shape* lastReflection, Intersection& closestIntersection) {
        Intersection intersections[10];
        bool hitAtLeastOneObject = false;
        float minDist = 100000000;
        
        closestIntersection.shape = NULL;
        
        int count = 0;
        
        for(Triangle* shape = scene.triangles.begin(); shape != scene.triangles.end(); ++shape) {
            if(shape == lastReflection)
                continue;
            
            int totalIntersections = shape->calculateRayIntersections(ray, intersections);
            count += totalIntersections;
            
            findClosestIntersection(ray.v[0], closestIntersection, intersections, totalIntersections, minDist);
        }
        
        return closestIntersection.shape != NULL;
    }
    
    Color traceRay(const Ray& ray, int depth, const Shape* lastReflection) {
        Color rayColor = Vec3(0, 0, 0);
        Intersection closestIntersection;
        
        
        bool hitAtLeastOneObject = findNearestRayIntersection(ray, lastReflection, closestIntersection);
        
        if(hitAtLeastOneObject) {
            if(closestIntersection.shape->material.reflective) {
                if(depth < 5) {
                    Ray reflectedRay = ray.reflectAboutNormal(closestIntersection.normal, closestIntersection.pos);
                    
                    Color reflectedRayColor = traceRay(reflectedRay, depth + 1, closestIntersection.shape);
                    
                    float dot = -ray.dir.dot(closestIntersection.normal);
                    float alpha = .9;
                    float fresnelEffect = pow(1 - dot, 3) * (1 - alpha) + 1 * alpha;
                    
                    rayColor = (reflectedRayColor * fresnelEffect + calculateOnlySpecularHightlights(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos)).maxValue(1.0);
                }
            }
            else {
                rayColor = calculateLighting(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos);
            }
        }
        
        return rayColor;
    }
    
    void findClosestIntersection(Vec3 start, Intersection& closestIntersection, Intersection* intersections, int totalIntersections, float& minDist) {
        for(int i = 0; i < totalIntersections; ++i) {
            float dist = start.distanceBetween(intersections[i].pos);
            
            if(dist < minDist) {
                minDist = dist;
                closestIntersection = intersections[i];
            }
        }
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

