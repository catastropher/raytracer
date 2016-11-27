#pragma once

#include "Ray.hpp"
#include "Material.hpp"

struct Shape;

struct Intersection {
    Vec3 pos;
    Vec3 normal;
    const Shape* shape;
};

struct Shape {
    Color color;
    Material material;
    
    virtual int calculateRayIntersections(const Ray& ray, Intersection* dest) const = 0;
    virtual Vec3 calculateNormalAtPoint(Vec3& point) const = 0;
    
    virtual ~Shape() { }
};

