#pragma once

#include "Vec3.hpp"

struct Ray {
    Vec3 v[2];
    Vec3 dir;
    
    CUDA_CALLABLE Ray(Vec3 v0, Vec3 v1) {
        v[0] = v0;
        v[1] = v1;
        
        dir = (v[1] - v[0]).normalize();
    }
    
    CUDA_CALLABLE Ray() { }
    
    CUDA_CALLABLE Vec3 calculatePointOnLine(float t) const {
        return v[0] + (v[1] - v[0]) * t;
    }
    
    CUDA_CALLABLE Ray reflectAboutNormal(const Vec3& normal, const Vec3& intersectionPoint) const {
        Vec3 newDir = dir.reflectAboutNormal(normal);
        return Ray(intersectionPoint, intersectionPoint + newDir);
    }
};

