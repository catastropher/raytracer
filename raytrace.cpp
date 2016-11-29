#include "config.hpp"

#include <iostream>
#include <cmath>
#include <assert.h>
#include <cstring>

#ifdef __WITH_SDL__
  #include "quickcg.h"
#endif

#include "Renderer.hpp"
#include "Sphere.hpp"
#include "Vec3.hpp"
#include "ModelLoader.hpp"

using namespace std;

#ifdef __WITH_SDL__
using namespace QuickCG;
#endif

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
    bool saveFrame = false;
    bool loadFrame = false;
    string frameFileName;
    int w = 640;
    int h = 480;
    
    for(int i = 1; i < argc; ++i) {
        if(strcmp(argv[i], "--save") == 0) {
            saveFrame = true;
            frameFileName = argv[++i];
        }
        else if(strcmp(argv[i], "--load") == 0) {
            loadFrame = true;
            frameFileName = argv[++i];
        }
        else if(strcmp(argv[i], "-w") == 0) {
            w = atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-h") == 0) {
            h = atoi(argv[++i]);
        }
    }
    
    Renderer renderer(60.0, w, h);
    
    renderer.ambientLightIntensity = .1;
    renderer.camPosition = Vec3(0, -100, 100);
    buildTestScene(renderer);
    
#ifdef __WITH_SDL__
    cls(RGB_Black);
#endif
    
    if(loadFrame) {
        renderer.loadFrameBuffer(frameFileName);
        renderer.displayFrameBuffer();
    }
    else {
        renderer.raytrace();
    }
    
    if(saveFrame) {
        renderer.saveFrameBuffer(frameFileName);
    }
    
#ifdef __WITH_SDL__
    while(!done()) {
        //++sp->center.z;
        redraw();
    }
#endif
}

