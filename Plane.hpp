#pragma once

#include "Vec3.hpp"

struct Plane {
    Vec3 normal;
    float d;
    
    CUDA_CALLABLE bool calculateRayIntersection(const Ray& ray, Vec3& intersectDest) const {
        float den = ray.dir.dot(normal);
        
        if(den == 0)
            return false;
        
        float t = -(ray.v[0].dot(normal) + d) / den;
        
        if(t < 0)
            return false;
        
        intersectDest = ray.v[0] + ray.dir * t;
        
        return true;
    }
    
    CUDA_CALLABLE Plane(const Vec3& p, const Vec3& u, const Vec3& v) {
        normal = u.cross(v).normalize();
        d = -p.dot(normal);
    }
    
    CUDA_CALLABLE Plane(float a, float b, float c, float d_) {
        normal.x = a;
        normal.y = b;
        normal.z = c;
        d = d_;
    }
    
    CUDA_CALLABLE Plane() { }
};

