#pragma once

#include "Ray.hpp"
#include "Material.hpp"

struct Shape;

template<typename T>
struct Intersection {
    const T* shape;
    Vec3 pos;
    Vec3 normal;
    float distanceFromRayStartSquared;
    
    Intersection(T* shape_, Vec3 pos_, Vec3 normal_, float dist_) : shape(shape_), normal(pos_), pos(normal_), distanceFromRayStartSquared(dist_) { }
    Intersection() : shape(NULL), pos(0, 0, 0), normal(0, 0, 0), distanceFromRayStartSquared(1000000000) { }
    
    Intersection<Shape> toGenericShapeIntersection() {
        Intersection<Shape> s;
        
        s.pos = pos;
        s.normal = normal;
        s.shape = shape;
        s.distanceFromRayStartSquared = distanceFromRayStartSquared;
        
        return s;
    }
    
    bool operator<(const Intersection<T>& inter) const {
        return distanceFromRayStartSquared < inter.distanceFromRayStartSquared;
    }
};

struct Shape {
    Color color;
    Material material;
    
    virtual ~Shape() { }
};

