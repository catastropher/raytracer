#include <iostream>
#include <cmath>
#include "quickcg.h"

using namespace std;
using namespace QuickCG;

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
};

const float PI = 3.1415926535897;

float degToRadians(float deg) {
    return deg * PI / 180.0;
}


struct Ray {
    Vec3 v[2];
    
    Ray(Vec3 v0, Vec3 v1) {
        v[0] = v0;
        v[1] = v1;
    }
    
    Vec3 calculatePointOnLine(float t) const {
        return v[0] + (v[1] - v[0]) * t;
    }
};

struct Sphere {
    float radius;
    Vec3 center;
    
    // Returns the number of intersections found
    int calculateRayIntersections(const Ray& ray, Vec3* intersectDest) const {
        float xA = ray.v[0].x;
        float yA = ray.v[0].y;
        float zA = ray.v[0].z;
        
        float xB = ray.v[1].x;
        float yB = ray.v[1].y;
        float zB = ray.v[1].z;
        
        float xC = center.x;
        float yC = center.y;
        float zC = center.z;
        
        float a = pow(xB - xA, 2) + pow(yB - yA, 2) + pow(zB - zA, 2);
        float b = 2 * ((xB - xA) * (xA - xC) + (yB - yA) * (yA - yC) + (zB - zA) * (zA - zC));
        float c = pow(xA - xC, 2) + pow(yA - yC, 2) + pow(zA - zC, 2) - pow(radius, 2);
        
        float delta = pow(b, 2) - 4 * a * c;
        
        int totalIntersections = 0;
        float d[2];
        
        if(delta < 0)
            return 0;
        
        if(delta == 0) {
            totalIntersections = 1;
            d[0] = -b / (2 * a);
        }
        else {
            totalIntersections = 2;
            d[0] = (-b + sqrt(delta)) / (2 * a);
            d[1] = (-b - sqrt(delta)) / (2 * a);
            
            //cout << d[0] << " " << d[1] << endl;
        }
        
        for(int i = 0; i < totalIntersections; ++i) {
            intersectDest[i] = ray.calculatePointOnLine(d[i]);
            
            cout << intersectDest[i].z << endl;
        }
        
        return totalIntersections;
    }
};

struct Renderer {
    float screenW, screenH;
    float viewAngle;
    float distToScreen;
    
    Renderer(float angle, float w, float h) {
        screen(640, 480);
        
        screenW = w;
        screenH = h;
        viewAngle = angle;
        distToScreen = (w / 2 ) / tan(degToRadians(angle / 2));
        
        cout << "Dist to screen: " << distToScreen << endl;
    }
    
    Ray calculateRayForPixelOnScreen(int x, int y) {
        Vec3 cameraPos(0, 0, 0);
        Vec3 pixelPos(x - w / 2, y - h / 2, distToScreen);
        
        return Ray(cameraPos, pixelPos);
    }
    
    void raytrace() {
        Sphere sphere;
        
        sphere.radius = 500;
        sphere.center = Vec3(0, 0, 2000);
        
        for(int i = 0; i < screenH; ++i) {
            for(int j = 0; j < screenW; ++j) {
                Ray ray = calculateRayForPixelOnScreen(j, i);
                Vec3 intersections[2];
                
                int total = sphere.calculateRayIntersections(ray, intersections);
                
                if(total > 0) {
                    Vec3 closest;
                    
                    if(total == 1) {
                        closest = intersections[0];
                    }
                    else {
                        closest = (intersections[0].z < intersections[1].z ? intersections[0] : intersections[1]);
                    }
                    
                    float minDist = 10;
                    float maxDist = 1000;
                    
                    float intensity = 1.0 - (closest.z - (sphere.center.z - sphere.radius)) / sphere.radius; //((maxDist - minDist) - (closest.z - minDist)) / (maxDist - minDist);
                    
                    ColorRGB color(255 * intensity, 0, 0);
                    
                    pset(j, i, color);
                }
            }
        }
    }
};














int main(int argc, char* argv[]) {    
    Renderer renderer(60.0, 640, 480);
    
    cls(RGB_Black);
    
    renderer.raytrace();
    
    while(!done()) {
        redraw();
    }
}


