#pragma once

#include "Material.hpp"

struct Light {
    Color color;
    Vec3 pos;
    float intensity;
    Vec3 dir;
    
    Color calculateSpecular(const Material& mat, const Vec3& pointOnObj, const Vec3& camPos, const Vec3& objNormal) {
        Vec3 L = (pos - pointOnObj).normalize();
        Vec3 R = L.neg().reflectAboutNormal(objNormal);
        Vec3 V = (camPos - pointOnObj).normalize();
        
        return Color(1.0, 1.0, 1.0) * pow(std::max(0.0f, V.dot(R)), mat.alpha) * mat.specular * intensity;
    }
    
    Color evaluatePhongReflectionModel(const Material& mat, const Color& objColor, const Vec3& objNormal, const Vec3& pointOnObj, const Vec3& camPos) {
        Vec3 L = (pos - pointOnObj).normalize();
        const Vec3& N = objNormal;
        
        float cosTheta = std::max(0.0f, L.dot(N));// * (L.dot(N));
        Color diffuseColor = Vec3(0, 0, 0);
        
        if(cosTheta > 0)
            diffuseColor = color.multiplyEach(objColor) * cosTheta * mat.diffuse * intensity;
        
        Color specularColor = calculateSpecular(mat, pointOnObj, camPos, objNormal);
        
        return diffuseColor + specularColor;
    }
    
    void lookAt(Vec3 posToLookAt) {
        dir = (posToLookAt - pos).normalize();
    }
};

