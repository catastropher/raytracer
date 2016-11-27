#include <iostream>
#include <cmath>
#include <assert.h>

#include "quickcg.h"
#include "Renderer.hpp"
#include "Sphere.hpp"
#include "Vec3.hpp"
#include "ModelLoader.hpp"

using namespace std;
using namespace QuickCG;

Sphere* sp;

void buildTestScene(Renderer& renderer) {
    Material mat;
    
    mat.alpha = 100.0;
    mat.diffuse = .5;
    mat.specular = 1.0;
    mat.reflective = true;
    
    
    Sphere* sphere = new Sphere();
        
    sp = sphere;
    
    sphere->radius = 20;
    sphere->center = Vec3(-100, -20, 400);
    sphere->color = Vec3(1, 0, 0);
    sphere->material = mat;
    
    mat.reflective = false;
    
    Sphere* sphere2 = new Sphere();
        
    sphere2->radius = 20;
    sphere2->center = Vec3(-25, -20, 400 - 75);
    sphere2->color = Vec3(1, 0, 0);
    sphere2->material = mat;
    
    
    float w = 400, h = 5000;
    float y = 0;
    float d = -1000;
    
    Vec3 v[4] = {
        Vec3(-w, y, d),
        Vec3(-w, y, d + h),
        Vec3(w, y, d + h),
        Vec3(w, y, d)
    };
    
    Vec3 center = Vec3(0, 0, 0);
    
    for(int i = 0; i < 4; ++i) {
        center = center +  v[i] * (1.0 / 4);
    }

    
    renderer.addQuad(v, mat, Vec3(0, 0, 1.0));
    
    
    renderer.addObjectToScene(sphere);
    renderer.addObjectToScene(sphere2);
    
    Light light;
    light.color = Vec3(1, 1, 1);
    light.pos = Vec3(-500, -300, -100);
    light.lookAt(center);
    light.intensity = .7;
    
    Light light2 = light;
    light2.pos.x = -light2.pos.x;
    light2.lookAt(center);
    
    ModelLoader loader;
    
    try {
        vector<Triangle> triangles = loader.loadFile("../teapot.obj");
        renderer.addTriangle(triangles);
    }
    catch(string s) {
        cerr << "Error: " << s << endl;
        exit(-1);
    }
    
    renderer.addLight(light);
    //renderer.addLight(light2);
}

int main(int argc, char* argv[]) {    
    Renderer renderer(60.0, 1024, 768);
    
    renderer.ambientLightIntensity = .1;
    
    renderer.camPosition = Vec3(0, -100, 100);
    
    cls(RGB_Black);
    
    buildTestScene(renderer);
    
    renderer.raytrace();
    
    while(!done()) {
        //++sp->center.z;
        redraw();
    }
}

