#include "config.hpp"

#include <iostream>
#include <cmath>
#include <assert.h>
#include <cstring>
#include <clocale>

#ifdef __WITH_SDL__
  #include "quickcg.h"
#endif

#ifdef __WITH_CUDA__
    #include "Cuda.hpp"
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

void buildTestScene(SceneBuilder& builder) {
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
    
    mat.reflective = true;
    
    Sphere* sphere2 = new Sphere();
        
    
    Material sphereMaterial = mat;
    sphereMaterial.reflective = false;
    sphere2->radius = 20;
    sphere2->center = Vec3(-100, -150, 400);
    sphere2->color = Vec3(1, 0, 0);
    sphere2->material = sphereMaterial;
    
    
    float w = 800, h = 5000;
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
    
    float d2 = 800;
    float h2 = -1000;
    float w2 = 2000;
    Vec3 v2[4] = {
        Vec3(-w2, y, d2),
        Vec3(-w2, y + h2, d2),
        Vec3(w2, y + h2, d2),
        Vec3(w2, y, d2)
    };

    Material floor = mat;
    floor.reflective = true;
    floor.specular = 0;
    builder.addQuad(v, floor, Vec3(0, 0, 1.0));

    builder.addQuad(v2, floor, Vec3(0, 0, 1.0));
    
    
    builder.addSphere(*sphere);
    builder.addSphere(*sphere2);
    
    Light light;
    light.color = Vec3(1, 1, 1);
    light.pos = Vec3(-500, -300, -100);
    light.lookAt(center);
    light.intensity = .7;
    
    Light light2 = light;
    light2.pos.x = -light2.pos.x;
    light2.lookAt(center);
    
    ModelLoader pumpkinLoader;
    ModelLoader carLoader;
    
    Model pumpkin = pumpkinLoader.loadFile("../objects/pumpkin_tall_10k.obj");
    pumpkin.color(Color(1.0, .54, 0)).transform(
        Mat4::identity().translate(Vec3(-50, -50, 500)).rotateAroundX(90)
    );
    builder.addTriangles(pumpkin.triangles);
    
    Model car = carLoader.loadFile("../objects/alfa147.obj");
    car.color(Color(0, 0, 1)).transform(
        Mat4::identity().translate(Vec3(-200, -50, 600)).rotateAroundY(45).rotateAroundX(90)
    );
    builder.addTriangles(car.triangles);
    
    
    ModelLoader teapotLoader;
    
    Model teapot = teapotLoader.loadFile("../objects/teapot.obj");
    teapot.flipUpsideDown().transform(Mat4::identity().translate(Vec3(100, -50, 400)));
    builder.addTriangles(teapot.triangles);
    
    builder.addLight(light);
    //renderer.addLight(light2);
}

int main(int argc, char* argv[]) {
    try {
        bool saveFrame = false;
        bool loadFrame = false;
        string frameFileName;
        int w = 640;
        int h = 480;    
        bool wait = true;
        
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
            else if(strcmp(argv[i], "--nowait") == 0) {
                wait = false;
            }
        }
        
        RayTracer<CPURayTracer> tracer;
        tracer.renderer.initialize(60.0, w, h);
        
        SceneBuilder builder;
        
        builder.ambientLightIntensity = .1;
        builder.camPosition = Vec3(0, -100, 100);
        buildTestScene(builder);
        
        tracer.scene = builder.buildScene();
        
    #ifdef __WITH_CUDA__
        printf("Invoking CUDA kernel\n");
        launchCudaKernel(60.0, w, h, tracer.scene, tracer.renderer);
    #endif
        
    #ifdef __WITH_SDL__
        cls(RGB_Black);
    #endif
        
        if(loadFrame) {
            tracer.renderer.loadFrameBuffer(frameFileName);
            tracer.renderer.displayFrameBuffer();
        }
        else {
    #ifndef __WITH_CUDA__
            tracer.raytrace();
    #endif
        }
        
        if(saveFrame) {
            tracer.renderer.saveFrameBuffer(frameFileName);
        }
        
    #ifdef __WITH_SDL__
        redraw();
        
        setlocale(LC_NUMERIC, "");
        printf("There are %'lld triangles in the scene\n", (unsigned long long)builder.triangles.size());
        printf("Required %'lld triangle intersections to render\n", tracer.tracer.triangleIntersectionCount);
        
        if(wait) {
            while(!done()) {
                //++sp->center.z;
            }
        }
    #endif
    }
    catch(string s) {
        cerr << "Error: " << s << endl;
        exit(-1);
    }
    catch(const char* s) {
        cerr << "Error: " << s << endl;
        exit(-1);
    }
}

