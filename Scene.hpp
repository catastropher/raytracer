#pragma once

#include <vector>

#include "Vec3.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"
#include "Light.hpp"

template<typename T>
struct GeometryList {
    T* list;
    int total;
    
    T* begin() const {
        return list;
    }
    
    T* end() const {
        return list + total;
    }
};

struct Scene {    
    Vec3 camPosition;
    
    GeometryList<Triangle> triangles;
    GeometryList<Sphere> spheres;
    GeometryList<Light> lights;
    
    float ambientLightIntensity;
};

struct SceneBuilder {
    std::vector<Light> lightsInScene;
    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;
    
    float ambientLightIntensity;
    Vec3 camPosition;
    
    void addLight(Light light) {
        lightsInScene.push_back(light);
    }
    
    void addTriangle(Triangle tri) {
        triangles.push_back(tri);
    }
    
    void addSphere(Sphere sphere) {
        spheres.push_back(sphere);
    }
    
    void addTriangles(std::vector<Triangle>& tri) {
        triangles.insert(triangles.end(), tri.begin(), tri.end());
    }
    
    void addQuad(Vec3 v[4], Material& mat, Color color) {
        Triangle t1(v[0], v[2], v[0]);
        t1.material = mat;
        t1.color = color;
        
        Triangle t2(v[2], v[0], v[3]);
        t2.material = mat;
        t2.color = color;
        
        addTriangle(t1);
        addTriangle(t2);
    }
    
    Scene buildScene() {
        Scene scene;
        
        scene.triangles.list = new Triangle[triangles.size()];
        scene.triangles.total = triangles.size();
        
        for(int i = 0; i < triangles.size(); ++i) {
            scene.triangles.list[i] = triangles[i];
        }
        
        scene.spheres.list = new Sphere[spheres.size()];
        scene.spheres.total = triangles.size();
        
        for(int i = 0; i < spheres.size(); ++i) {
            scene.spheres.list[i] = spheres[i];
        }
        
        scene.lights.list = new Light[triangles.size()];
        scene.lights.total = lightsInScene.size();
        
        for(int i = 0; i < lightsInScene.size(); ++i) {
            scene.lights.list[i] = lightsInScene[i];
        }
        
        scene.camPosition = camPosition;
        scene.ambientLightIntensity = ambientLightIntensity;
        
        return scene;
    }
};

