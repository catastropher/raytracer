#pragma once

#include "Vec3.hpp"

struct Ray {
    Vec3 v[2];
    Vec3 dir;
    
    Ray(Vec3 v0, Vec3 v1) {
        v[0] = v0;
        v[1] = v1;
        
        dir = (v[1] - v[0]).normalize();
    }
    
    Ray() { }
    
    Vec3 calculatePointOnLine(float t) const {
        return v[0] + (v[1] - v[0]) * t;
    }
    
    Ray reflectAboutNormal(const Vec3& normal, const Vec3& intersectionPoint) const {
        Vec3 newDir = dir.reflectAboutNormal(normal);
        return Ray(intersectionPoint, intersectionPoint + newDir);
    }
};

