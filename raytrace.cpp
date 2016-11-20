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
    
    Vec3 cross(const Vec3& v) const {
        const Vec3& u = *this;
        
        return Vec3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x
        );
    }
    
    float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    string toString() const {
        char str[128];
        sprintf(str, "{ %f, %f, %f }", x, y, z);
        
        return string(str);
    }
};

const float PI = 3.1415926535897;

float degToRadians(float deg) {
    return deg * PI / 180.0;
}

struct Ray {
    Vec3 v[2];
    Vec3 dir;
    
    Ray(Vec3 v0, Vec3 v1) {
        v[0] = v0;
        v[1] = v1;
        
        dir = (v[1] - v[0]).normalize();
    }
    
    Vec3 calculatePointOnLine(float t) const {
        return v[0] + (v[1] - v[0]) * t;
    }
};

struct Shape {
    virtual int calculateRayIntersections(const Ray& ray, Vec3* intersectDest) const = 0;
    virtual Vec3 calculateNormalAtPoint(Vec3& point) const = 0;
    
    virtual ~Shape() { }
};

struct Plane {
    Vec3 normal;
    float d;
    
    bool calculateRayIntersection(const Ray& ray, Vec3& intersectDest) const {
        float den = ray.dir.dot(normal);
        
        if(den == 0)
            return false;
        
        float t = -(ray.v[0].dot(normal) + d) / den;
        intersectDest = ray.v[0] + ray.dir * t;
        
        return true;
    }
    
    Plane(const Vec3& p, const Vec3& u, const Vec3& v) {
        normal = u.cross(v).normalize();
        d = -p.dot(normal);
        
        cout << "Normal: " << normal.toString() << ", " << d << endl;
    }
    
    Plane() { }
};


void line(Vec3 v0, Vec3 v1) {
    drawLine(v0.x + 320, v0.y + 240, v1.x + 320, v1.y + 240, RGB_Blue);
}

struct Triangle : Shape {
    Vec3 p[3];
    Plane plane;
    
    Triangle(Vec3 p0, Vec3 p1, Vec3 p2) {
        p[0] = p0;
        p[1] = p1;
        p[2] = p2;
        
        
        line(p[0], p[1]);
        line(p[1], p[2]);
        line(p[0], p[2]);
        
        Vec3 u = (p[1] - p[0]).normalize();
        Vec3 v = (p[2] - p[0]).normalize();
        plane = Plane(p[0], u, v);
    }
    
    int calculateRayIntersections(const Ray& ray, Vec3* intersectDest) const {
        Vec3 u = (p[1] - p[0]);
        Vec3 v = (p[2] - p[0]);
        
        if(!plane.calculateRayIntersection(ray, *intersectDest))
            return 0;
        
        //cout << intersectDest.toString() << endl;
        
        Vec3 w = (*intersectDest - p[0]);
        
        float uv = u.dot(v);
        float uu = u.dot(u);
        
        float vv = v.dot(v);
        
        float wv = w.dot(v);
        float wu = w.dot(u);
        
        float d = uv * uv - uu * vv;
        float s1 = (uv * wv - vv * wu) / d;
        float t1 = (uv * wu - uu * wv) / d;
        
        if(s1 < 0 || s1 > 1.0 || t1 < 0 || (s1 + t1) > 1.0)
            return 0;
        
        return 1;
    }
    
    Vec3 calculateNormalAtPoint(Vec3& point) const {
        
    }
};

struct Sphere : Shape {
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
    
    Vec3 calculateNormalAtPoint(Vec3& point) const {
        
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
        
        float w = 200, h = 1000;
        float y = 200;
        float d = 500;
        
        Vec3 v[4] = {
            Vec3(-w, y, d),
            Vec3(-w, y, d + h),
            Vec3(w, y, d + h),
            Vec3(w, y, d)
        };
        
        
        Triangle tri(
            v[0],
            v[1],
            v[2]
        );
        
        Triangle tri2(
            v[2], v[3], v[0]
        );
        
        for(int i = 0; i < screenH; ++i) {
            for(int j = 0; j < screenW; ++j) {
                Ray ray = calculateRayForPixelOnScreen(j, i);
                Vec3 intersections[2];
                
                if(i == 240 - 50 && j == 320)
                    cout << ray.dir.toString() << endl;
                
                int total = tri.calculateRayIntersections(ray, &intersections[0]) +  //sphere.calculateRayIntersections(ray, intersections);
                    tri2.calculateRayIntersections(ray, &intersections[1]);
                
                
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
                    
                    float intensity = 1.0;//1.0 - (closest.z - (sphere.center.z - sphere.radius)) / sphere.radius; //((maxDist - minDist) - (closest.z - minDist)) / (maxDist - minDist);
                    
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


