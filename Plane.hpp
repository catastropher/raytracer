#pragma once

#include "Vec3.hpp"

struct Plane {
    Vec3 normal;
    float d;
    
    bool calculateRayIntersection(const Ray& ray, Vec3& intersectDest) const {
        float den = ray.dir.dot(normal);
        
        if(den == 0)
            return false;
        
        float t = -(ray.v[0].dot(normal) + d) / den;
        
        if(t < 0)
            return false;
        
        intersectDest = ray.v[0] + ray.dir * t;
        
        return true;
    }
    
    Plane(const Vec3& p, const Vec3& u, const Vec3& v) {
        normal = u.cross(v).normalize();
        d = -p.dot(normal);
    }
    
    Plane() { }
};

