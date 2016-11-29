#pragma once

#include <string>
#include <cmath>
#include <algorithm>

#include "config.hpp"

struct Vec3 {
    float x, y, z;
    
    CUDA_CALLABLE Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) { }
    CUDA_CALLABLE Vec3() : x(0), y(0), z(0) { }
    
    CUDA_CALLABLE Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    
    CUDA_CALLABLE Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    
    CUDA_CALLABLE float lengthSquared() const {
        return x * x + y * y + z * z;
    }
    
    CUDA_CALLABLE float length() const {
        return sqrt(lengthSquared());
    }
    
    CUDA_CALLABLE Vec3 normalize() const {
        float len = length();
        
        return Vec3(x / len, y / len, z / len);
    }
    
    CUDA_CALLABLE Vec3 operator*(float f) const {
        return Vec3(x * f, y * f, z * f);
    }
    
    CUDA_CALLABLE Vec3 cross(const Vec3& v) const {
        const Vec3& u = *this;
        
        return Vec3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x
        );
    }
    
    CUDA_CALLABLE Vec3 neg() const {
        return Vec3(-x, -y, -z);
    }
    
    CUDA_CALLABLE Vec3 multiplyEach(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }
    
    CUDA_CALLABLE Vec3 maxValue(float maxValueAllowed) {
        return Vec3(std::min(maxValueAllowed, x), std::min(maxValueAllowed, y), std::min(maxValueAllowed, z));
    }
    
    CUDA_CALLABLE Vec3 reflectAboutNormal(const Vec3& normal) const {
        float ndot = -normal.dot(*this);
        return (*this + (normal * 2 * ndot)).normalize();
    }
    
    CUDA_CALLABLE float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    CUDA_CALLABLE float distanceBetween(Vec3& v) {
        return (*this - v).length();
    }
    
    std::string toString() const {
        char str[128];
        sprintf(str, "{ %f, %f, %f }", x, y, z);
        
        return std::string(str);
    }
};

