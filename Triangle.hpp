#pragma once

#include "Plane.hpp"
#include "Shape.hpp"

struct CudaTriangleAttributes : Shape {
    Vec3 normals[3];
    
    CUDA_CALLABLE Vec3 calculateNormal(float u, float v) const {
        return normals[0] * (1.0 - u - v) + normals[1] * u + normals[2] * v;
    }
};

struct CudaTriangle {
    float data[0];
    
    CUDA_CALLABLE Vec3 getVertex(int vertex) const {
        return Vec3(
            data[vertex * 3 + 0],
            data[vertex * 3 + 1],
            data[vertex * 3 + 2]
        );
    }
    
    CUDA_CALLABLE void setVertex(int vertex, Vec3 v) {
        data[vertex * 3 + 0] = v.x;
        data[vertex * 3 + 1] = v.y;
        data[vertex * 3 + 2] = v.z;
    }
    
    CUDA_CALLABLE Plane getPlane() const {
        return Plane(data[9], data[10], data[11], data[12]);
    }
    
    void setPlane(Plane p) {
        data[9] = p.normal.x;
        data[10] = p.normal.y;
        data[11] = p.normal.z;
        data[12] = p.d;
    }
    
    CUDA_CALLABLE int calculateRayIntersections(const Ray& ray, CudaTriangleIntersection* intersectDest) {        
        Vec3 p[3] = {
            getVertex(0),
            getVertex(1),
            getVertex(2)
        };
        
        Vec3 u = (p[1] - p[0]);
        Vec3 v = (p[2] - p[0]);
        
        Plane plane = getPlane();
        
        if(!plane.calculateRayIntersection(ray, intersectDest->pos))
            return 0;
        
        if(ray.dir.neg().dot(plane.normal) < 0)
            return 0;
        
        Vec3 w = (intersectDest->pos - p[0]);
        
        float uv = u.dot(v);
        float uu = u.dot(u);
        
        float vv = v.dot(v);
        
        float wv = w.dot(v);
        float wu = w.dot(u);
        
        float d = uv * uv - uu * vv;
        
        if(d == 0)
            return 0;
        
        float s1 = (uv * wv - vv * wu) / d;
        float t1 = (uv * wu - uu * wv) / d;
        
        if(s1 < 0 || s1 > 1.0 || t1 < 0 || (s1 + t1) > 1.0)
            return 0;
        
        intersectDest->triangleStart = data;
        intersectDest->distanceFromRayStartSquared = (intersectDest->pos - ray.v[0]).lengthSquared();
        intersectDest->intersectionS = s1;
        intersectDest->intersectionT = t1;
        
        return 1;
    }
};

struct Triangle : Shape {
    Vec3 p[3];
    Plane plane;
    Vec3 normals[3];
    
    CUDA_CALLABLE Triangle() { }
    
    CUDA_CALLABLE Triangle(Vec3 p0, Vec3 p1, Vec3 p2) {
        p[0] = p0;
        p[1] = p1;
        p[2] = p2;
        
        Vec3 u = (p[1] - p[0]).normalize();
        Vec3 v = (p[2] - p[0]).normalize();
        plane = Plane(p[0], u, v);
        
        normals[0] = plane.normal;
        normals[1] = plane.normal;
        normals[2] = plane.normal;
    }
    
    CUDA_CALLABLE int calculateRayIntersections(const Ray& ray, Intersection<Triangle>* intersectDest) const {
        Vec3 u = (p[1] - p[0]);
        Vec3 v = (p[2] - p[0]);
        
        if(!plane.calculateRayIntersection(ray, intersectDest->pos))
            return 0;
        
        //cout << intersectDest.toString() << endl;
        
        if(ray.dir.neg().dot(plane.normal) < 0)
            return 0;
        
        Vec3 w = (intersectDest->pos - p[0]);
        
        float uv = u.dot(v);
        float uu = u.dot(u);
        
        float vv = v.dot(v);
        
        float wv = w.dot(v);
        float wu = w.dot(u);
        
        float d = uv * uv - uu * vv;
        
        if(d == 0)
            return 0;
        
        float s1 = (uv * wv - vv * wu) / d;
        float t1 = (uv * wu - uu * wv) / d;
        
        if(s1 < 0 || s1 > 1.0 || t1 < 0 || (s1 + t1) > 1.0)
            return 0;
        
        intersectDest->normal = calculateNormal(s1, t1); //calculateNormalAtPoint(intersectDest->pos);
        intersectDest->shape = this;
        intersectDest->distanceFromRayStartSquared = (intersectDest->pos - ray.v[0]).lengthSquared();
        
        return 1;
    }
    
    CUDA_CALLABLE void setNormals(Vec3 n0, Vec3 n1, Vec3 n2) {
        normals[0] = n0;
        normals[1] = n1;
        normals[2] = n2;
    }
    
    CUDA_CALLABLE Vec3 calculateNormal(float u, float v) const {
        return normals[0] * (1.0 - u - v) + normals[1] * u + normals[2] * v;
    }
    
    CUDA_CALLABLE Vec3 calculateNormalAtPoint(Vec3& point) const {
        return plane.normal;
    }
};
