#pragma once

#include <cmath>

#include "config.hpp"
#include "Vec3.hpp"

struct Mat4 {
    float elements[4][4];
    
    Mat4(const float e[4][4]) {
        for(int i = 0; i < 4; ++i)
            for(int j = 0; j < 4; ++j)
                elements[i][j] = e[i][j];
    }
    
    Mat4() { }
    
    Mat4 multiply(Mat4 mat) const {
        Mat4 res;
        
        for(int i = 0; i < 4; ++i) {
            for(int j = 0; j < 4; ++j) {
                float sum = 0;
                
                for(int k = 0; k < 4; ++k) {
                    sum += elements[i][k] * mat.elements[k][j];
                }
                
                res.elements[i][j] = sum;
            }
        }
        
        return res;
    }
    
    Mat4 rotateAroundY(float angle) const {
        float sinAngle = sin(degToRadians(angle));
        float cosAngle = cos(degToRadians(angle));
        
        float mat[4][4] = {
            { cosAngle, 0,  -sinAngle,  0 },
            { 0,        1,  0,          0 },
            { sinAngle, 0,  cosAngle,   0 },
            { 0,        0,  0,          1 }
        };
        
        return multiply(Mat4(mat));
    }
    
    Mat4 rotateAroundX(float angle) const {
        float sinAngle = sin(degToRadians(angle));
        float cosAngle = cos(degToRadians(angle));
        
        float mat[4][4] = {
            { 1, 0,        0,           0 },
            { 0, cosAngle, -sinAngle,   0 },
            { 0, sinAngle, cosAngle,    0 },
            { 0, 0,        0,           1 }
        };
        
        return multiply(Mat4(mat));
    }
    
    Mat4 translate(const Vec3 v) const {
        float mat[4][4] = {
            { 1, 0, 0, v.x },
            { 0, 1, 0, v.y },
            { 0, 0, 1, v.z },
            { 0, 0, 0, 1   }
        };
        
        return multiply(Mat4(mat));
    }
    
    Vec3 transformVec4(const float* vv) const {
        float rot[4];
        
        for(int i = 0; i < 4; ++i) {
            rot[i] = 0;
            
            for(int j = 0; j < 4; ++j) {
                rot[i] += vv[j] * elements[i][j];
            }
        }
        
        float w = rot[3];
        
        if(w != 0)
            return Vec3(rot[0] / w, rot[1] / w, rot[2] / w);
        else
            return Vec3(rot[0], rot[1], rot[2]);
    }
    
    Vec3 rotateVec3(const Vec3 v) const {
        float vv[4] = { v.x, v.y, v.z, 1 };
        return transformVec4(vv);
    }
    
    Vec3 rotateVec3Normal(const Vec3 v) const {
        Mat4 mat = identity();
        
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                mat.elements[i][j] = elements[j][i];
        
        float vv[4] = { v.x, v.y, v.z, 0 };
        return transformVec4(vv);
    }
    
    static Mat4 identity() {
        const float mat[4][4] = {
            { 1, 0, 0, 0 },
            { 0, 1, 0, 0 },
            { 0, 0, 1, 0 },
            { 0, 0, 0, 1 }
        };
        
        return Mat4(mat);
    }
};

