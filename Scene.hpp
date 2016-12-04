#pragma once

#include <vector>

#include "Vec3.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"
#include "Light.hpp"

template<typename T>
struct GeometryList {
    T* list;
    size_t total;
    
    CUDA_CALLABLE T* begin() const {
        return list;
    }
    
    CUDA_CALLABLE T* end() const {
        return list + total;
    }
    
    CUDA_CALLABLE GeometryList() {
        list = NULL;
        total = 0;
    }
    
    CUDA_CALLABLE GeometryList(size_t total_) : total(total_) {
        list = (T *)malloc(sizeof(T) * total);
    }
    
    void cleanup() {
        free(list);
        list = NULL;
        total = 0;
    }
};



struct CUDATriangleList {
    GeometryList<float>                     triangleGeometry;
    GeometryList<CudaTriangleAttributes>    triangleAttributes;
};

static inline float& cudaTriangleListVX(float* t, int vertexId) {
    return t[vertexId * 3 + 0];
}

static inline float& cudaTriangleListVY(float* t, int vertexId) {
    return t[vertexId * 3 + 1];
}

static inline float& cudaTriangleListVZ(float* t, int vertexId) {
    return t[vertexId * 3 + 2];
}

static inline float& cudaTriangleListPlaneA(float* t) {
    return t[3 * 3 + 0];
}

static inline float& cudaTriangleListPlaneB(float* t) {
    return t[3 * 3 + 1];
}

static inline float& cudaTriangleListPlaneC(float* t) {
    return t[3 * 3 + 2];
}

static inline float& cudaTriangleListPlaneD(float* t) {
    return t[3 * 3 + 3];
}

const int cudaTriangleListSize = 13;

struct Scene {    
    Vec3 camPosition;
    
    GeometryList<Triangle> triangles;
    GeometryList<Sphere> spheres;
    GeometryList<Light> lights;
    
//#ifdef __WITH_CUDA__
    CUDATriangleList cudaTriangles;
//#endif
    
    
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
        
//#ifdef __WITH_CUDA__
#if 1
        if((triangles.size() % THREADS_IN_BLOCK) != 0) {
            int trianglesToAdd = THREADS_IN_BLOCK - (triangles.size() % THREADS_IN_BLOCK);
            
            printf("Need to add %d triangles\n", trianglesToAdd);
            
            for(int i = 0; i < trianglesToAdd; ++i) {
                Vec3 farFarAway(0, 0, 100000000.0f);
                Vec3 farFarAway2(1, 0, 100000000.0f);
                Vec3 farFarAway3(0, 1, 100000000.0f);
                
                Triangle t(farFarAway, farFarAway2, farFarAway3);
                t.material.reflective = false;
                t.color = Vec3(0, 0, 0);
                t.material.alpha = 0;
                t.material.diffuse = 0;
                t.material.specular = 0;
                
                addTriangle(t);
            }
        }
#endif
//#endif
        
        scene.triangles.list = new Triangle[triangles.size()];
        scene.triangles.total = triangles.size();
        
        for(int i = 0; i < triangles.size(); ++i) {
            scene.triangles.list[i] = triangles[i];
        }
        
        scene.spheres.list = new Sphere[spheres.size()];
        scene.spheres.total = spheres.size();
        
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

