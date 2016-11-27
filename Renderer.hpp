#pragma once

#include <vector>
#include <iostream>

#include "Material.hpp"
#include "Light.hpp"
#include "Ray.hpp"
#include "Shape.hpp"
#include "Triangle.hpp"

#ifdef __WITH_SDL__
  #include "quickcg.h"
#endif

struct Renderer {
    float screenW, screenH;
    float viewAngle;
    float distToScreen;
    
    std::vector<Shape* > shapesInScene;
    
    float ambientLightIntensity;
    std::vector<Light> lightsInScene;
    
    Vec3 camPosition;
    
    Renderer(float angle, float w, float h) {
#ifdef __WITH_SDL__
        QuickCG::screen(w, h);
#endif
        
        screenW = w;
        screenH = h;
        viewAngle = angle;
        distToScreen = (w / 2 ) / tan(degToRadians(angle / 2));
        
        std::cout << "Dist to screen: " << distToScreen << std::endl;
        
        camPosition = Vec3(0, 0, 0);
    }
    
    float degToRadians(float deg) {
        const float PI = 3.1415926535897;
        return deg * PI / 180.0;
    }
    
    void addLight(Light light) {
        lightsInScene.push_back(light);
    }
    
    Ray calculateRayForPixelOnScreen(int x, int y) {
        Vec3 pixelPos(x - screenW / 2, y - screenH / 2, distToScreen);
        
        return Ray(camPosition, pixelPos + camPosition);
    }
    
    Color calculateOnlySpecularHightlights(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = Vec3(0, 0, 0);
        
        for(Light& light : lightsInScene) {
            result = result + light.calculateSpecular(shape->material, pointOnObj, camPosition, objNormal);
        }
        
        return result;
    }
    
    Color calculateLighting(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = shape->color * ambientLightIntensity;
        
        for(Light& light : lightsInScene) {
            Ray lightRay(pointOnObj, light.pos);
            Intersection closestIntersection;
            
            if(!findNearestRayIntersection(lightRay, shape, closestIntersection))
                result = result + light.evaluatePhongReflectionModel(shape->material, shape->color, objNormal, pointOnObj, camPosition);
        }
        
        return result.maxValue(1.0);
    }
    
    bool findNearestRayIntersection(const Ray& ray, const Shape* lastReflection, Intersection& closestIntersection) {
        Intersection intersections[10];
        bool hitAtLeastOneObject = false;
        float minDist = 100000000;
        
        closestIntersection.shape = NULL;
        
        int count = 0;
        
        for(Shape* shape : shapesInScene) {
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
    
    float distanceBetween(Vec3& v0, Vec3& v1) {
        return (v1 - v0).length();
    }
    
    void findClosestIntersection(Vec3 start, Intersection& closestIntersection, Intersection* intersections, int totalIntersections, float& minDist) {
        for(int i = 0; i < totalIntersections; ++i) {
            float dist = distanceBetween(start, intersections[i].pos);
            
            if(dist < minDist) {
                minDist = dist;
                closestIntersection = intersections[i];
            }
        }
    }
    
    void addObjectToScene(Shape* shape) {
        shapesInScene.push_back(shape);
    }
    
    void addTriangle(std::vector<Triangle> triangles) {
        for(Triangle t : triangles) {
            addObjectToScene(new Triangle(t));
        }
    }
    
    void addQuad(Vec3 v[4], Material& mat, Color color) {
        Triangle* t1 = new Triangle(v[0], v[2], v[0]);
        t1->material = mat;
        t1->color = color;
        
        Triangle* t2 = new Triangle(v[2], v[0], v[3]);
        t2->material = mat;
        t2->color = color;
        
        addObjectToScene(t1);
        addObjectToScene(t2);
    }
    
    void raytrace() {
        for(int i = 0; i < screenH; ++i) {
            for(int j = 0; j < screenW; ++j) {
                Ray ray = calculateRayForPixelOnScreen(j, i);
                
                Color rayColor = traceRay(ray, 0, NULL) * 255;
                
#ifdef __WITH_SDL__
                QuickCG::pset(j, i, QuickCG::ColorRGB(rayColor.x, rayColor.y, rayColor.z));
#endif
            }
           
#ifdef __WITH_SDL__
            QuickCG::redraw();
            QuickCG::redraw();
#endif
            std::cout << (i * 100 / screenH) << "%" << std::endl;
        }
    }
};