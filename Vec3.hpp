#pragma once

#include <string>
#include <cmath>
#include <algorithm>

struct Vec3 {
    float x, y, z;
    
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) { }
    Vec3() : x(0), y(0), z(0) { }
    
    Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    
    Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    
    float length() const {
        return sqrt(x * x + y * y + z * z);
    }
    
    Vec3 normalize() const {
        float len = length();
        
        return Vec3(x / len, y / len, z / len);
    }
    
    Vec3 operator*(float f) const {
        return Vec3(x * f, y * f, z * f);
    }
    
    Vec3 cross(const Vec3& v) const {
        const Vec3& u = *this;
        
        return Vec3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x
        );
    }
    
    Vec3 neg() const {
        return Vec3(-x, -y, -z);
    }
    
    Vec3 multiplyEach(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }
    
    Vec3 maxValue(float maxValueAllowed) {
        return Vec3(std::min(maxValueAllowed, x), std::min(maxValueAllowed, y), std::min(maxValueAllowed, z));
    }
    
    Vec3 reflectAboutNormal(const Vec3& normal) const {
        float ndot = -normal.dot(*this);
        return (*this + (normal * 2 * ndot)).normalize();
    }
    
    float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    std::string toString() const {
        char str[128];
        sprintf(str, "{ %f, %f, %f }", x, y, z);
        
        return std::string(str);
    }
};

