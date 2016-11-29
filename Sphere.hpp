#pragma once

#include "Shape.hpp"

struct Sphere : Shape {
    float radius;
    Vec3 center;
    
    // Returns the number of intersections found
    CUDA_CALLABLE int calculateRayIntersections(const Ray& ray, Intersection<Sphere>* intersectDest) const {
        float xA = ray.v[0].x;
        float yA = ray.v[0].y;
        float zA = ray.v[0].z;
        
        float xB = ray.v[1].x;
        float yB = ray.v[1].y;
        float zB = ray.v[1].z;
        
        float xC = center.x;
        float yC = center.y;
        float zC = center.z;
        
        float a = pow(xB - xA, 2) + pow(yB - yA, 2) + pow(zB - zA, 2);
        float b = 2 * ((xB - xA) * (xA - xC) + (yB - yA) * (yA - yC) + (zB - zA) * (zA - zC));
        float c = pow(xA - xC, 2) + pow(yA - yC, 2) + pow(zA - zC, 2) - pow(radius, 2);
        
        float delta = pow(b, 2) - 4 * a * c;
        
        int totalIntersections = 0;
        float d[2];
        
        if(delta < 0)
            return 0;
        
        if(delta == 0) {
            totalIntersections = 1;
            d[0] = -b / (2 * a);
        }
        else {
            totalIntersections = 2;
            d[0] = (-b + sqrt(delta)) / (2 * a);
            d[1] = (-b - sqrt(delta)) / (2 * a);
            
            //cout << d[0] << " " << d[1] << endl;
        }
        
        int validCount = 0;
        for(int i = 0; i < totalIntersections; ++i) {
            if(d[i] > 0) {
                intersectDest[validCount].pos = ray.calculatePointOnLine(d[i]);
                intersectDest[validCount].normal = calculateNormalAtPoint(intersectDest[i].pos);
                intersectDest[validCount].shape = this;
                intersectDest[validCount].distanceFromRayStartSquared = (intersectDest[validCount].pos - ray.v[0]).lengthSquared();
                ++validCount;
            }
        }
        
        return validCount;
    }
    
    CUDA_CALLABLE Vec3 calculateNormalAtPoint(Vec3& point) const {
        return (point - center).normalize();
    }
};

