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
    
    CUDA_CALLABLE Intersection(T* shape_, Vec3 pos_, Vec3 normal_, float dist_) : shape(shape_), normal(normal_), pos(pos_), distanceFromRayStartSquared(dist_) { }
    CUDA_CALLABLE Intersection() : shape(NULL), pos(0, 0, 0), normal(0, 0, 0), distanceFromRayStartSquared(1000000000) { }
    
    CUDA_CALLABLE Intersection<Shape> toGenericShapeIntersection() {
        Intersection<Shape> s;
        
        s.pos = pos;
        s.normal = normal;
        s.shape = shape;
        s.distanceFromRayStartSquared = distanceFromRayStartSquared;
        
        return s;
    }
    
    CUDA_CALLABLE bool operator<(const Intersection<T>& inter) const {
        return distanceFromRayStartSquared < inter.distanceFromRayStartSquared;
    }
};

struct CudaTriangleIntersection {
    float* triangleStart;
    float intersectionS;
    float intersectionT;
    float distanceFromRayStartSquared;
    Vec3 pos;
    
    CUDA_CALLABLE CudaTriangleIntersection() : triangleStart(NULL), pos(0, 0, 0), distanceFromRayStartSquared(1000000000) { }
    
    CUDA_CALLABLE bool operator<(const CudaTriangleIntersection& inter) const {
        return distanceFromRayStartSquared < inter.distanceFromRayStartSquared;
    }
};

struct Shape {
    Color color;
    Material material;
    
    CUDA_CALLABLE virtual ~Shape() { }
};

