#include <iostream>
#include <cmath>
#include <assert.h>

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
    
    Vec3 neg() const {
        return Vec3(-x, -y, -z);
    }
    
    Vec3 multiplyEach(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }
    
    Vec3 maxValue(float maxValueAllowed) {
        return Vec3(min(maxValueAllowed, x), min(maxValueAllowed, y), min(maxValueAllowed, z));
    }
    
    Vec3 reflectAboutNormal(const Vec3& normal) const {
        float ndot = -normal.dot(*this);
        return (*this + (normal * 2 * ndot)).normalize();
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

using Color = Vec3;

const float PI = 3.1415926535897;

float degToRadians(float deg) {
    return deg * PI / 180.0;
}

struct Material {
    float diffuse;
    float specular;
    float alpha;
    bool reflective;
    
    Material(float d, float s, float a, bool reflective_) : diffuse(d), specular(s), alpha(a), reflective(reflective_) { }
    Material() { }
};

struct Ray {
    Vec3 v[2];
    Vec3 dir;
    
    Ray(Vec3 v0, Vec3 v1) {
        v[0] = v0;
        v[1] = v1;
        
        dir = (v[1] - v[0]).normalize();
    }
    
    Ray() { }
    
    Vec3 calculatePointOnLine(float t) const {
        return v[0] + (v[1] - v[0]) * t;
    }
    
    Ray reflectAboutNormal(const Vec3& normal, const Vec3& intersectionPoint) const {
        Vec3 newDir = dir.reflectAboutNormal(normal);
        return Ray(intersectionPoint, intersectionPoint + newDir);
    }
};

struct Intersection;

struct Shape {
    Color color;
    Material material;
    
    virtual int calculateRayIntersections(const Ray& ray, Intersection* dest) const = 0;
    virtual Vec3 calculateNormalAtPoint(Vec3& point) const = 0;
    
    virtual ~Shape() { }
};

struct Intersection {
    Vec3 pos;
    Vec3 normal;
    const Shape* shape;
};

struct Plane {
    Vec3 normal;
    float d;
    
    bool calculateRayIntersection(const Ray& ray, Vec3& intersectDest) const {
        float den = ray.dir.dot(normal);
        
        if(den == 0)
            return false;
        
        float t = -(ray.v[0].dot(normal) + d) / den;
        
        if(t < 0)
            return false;
        
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
    //drawLine(v0.x + 320, v0.y + 240, v1.x + 320, v1.y + 240, RGB_Blue);
}

struct Triangle : Shape {
    Vec3 p[3];
    Plane plane;
    Vec3 normals[3];
    
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
        
        normals[0] = plane.normal;
        normals[1] = plane.normal;
        normals[2] = plane.normal;
    }
    
    int calculateRayIntersections(const Ray& ray, Intersection* intersectDest) const {
        Vec3 u = (p[1] - p[0]);
        Vec3 v = (p[2] - p[0]);
        
        if(!plane.calculateRayIntersection(ray, intersectDest->pos))
            return 0;
        
        //cout << intersectDest.toString() << endl;
        
        Vec3 w = (intersectDest->pos - p[0]);
        
        float uv = u.dot(v);
        float uu = u.dot(u);
        
        float vv = v.dot(v);
        
        float wv = w.dot(v);
        float wu = w.dot(u);
        
        float d = uv * uv - uu * vv;
        
        if(d == 0)
            return 0;
        
        float s1 = (uv * wv - vv * wu) / d;
        float t1 = (uv * wu - uu * wv) / d;
        
        if(s1 < 0 || s1 > 1.0 || t1 < 0 || (s1 + t1) > 1.0)
            return 0;
        
        intersectDest->normal = calculateNormal(s1, t1); //calculateNormalAtPoint(intersectDest->pos);
        intersectDest->shape = this;
        
        return 1;
    }
    
    void setNormals(Vec3 n0, Vec3 n1, Vec3 n2) {
        normals[0] = n0;
        normals[1] = n1;
        normals[2] = n2;
    }
    
    Vec3 calculateNormal(float u, float v) const {
        return normals[0] * (1.0 - u - v) + normals[1] * u + normals[2] * v;
    }
    
    Vec3 calculateNormalAtPoint(Vec3& point) const {
        return plane.normal;
    }
};

struct Sphere : Shape {
    float radius;
    Vec3 center;
    
    // Returns the number of intersections found
    int calculateRayIntersections(const Ray& ray, Intersection* intersectDest) const {
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
        
        int validCount = 0;
        for(int i = 0; i < totalIntersections; ++i) {
            if(d[i] > 0) {
                intersectDest[validCount].pos = ray.calculatePointOnLine(d[i]);
                intersectDest[validCount].normal = calculateNormalAtPoint(intersectDest[i].pos);
                intersectDest[validCount].shape = this;
                ++validCount;
            }
        }
        
        return validCount;
    }
    
    Vec3 calculateNormalAtPoint(Vec3& point) const {
        return (point - center).normalize();
    }
};

struct Light {
    Color color;
    Vec3 pos;
    float intensity;
    Vec3 dir;
    
    Color calculateSpecular(const Material& mat, const Vec3& pointOnObj, const Vec3& camPos, const Vec3& objNormal) {
        Vec3 L = (pos - pointOnObj).normalize();
        Vec3 R = L.neg().reflectAboutNormal(objNormal);
        Vec3 V = (camPos - pointOnObj).normalize();
        
        return Color(1.0, 1.0, 1.0) * pow(max(0.0f, V.dot(R)), mat.alpha) * mat.specular * intensity;
    }
    
    Color evaluatePhongReflectionModel(const Material& mat, const Color& objColor, const Vec3& objNormal, const Vec3& pointOnObj, const Vec3& camPos) {
        Vec3 L = (pos - pointOnObj).normalize();
        const Vec3& N = objNormal;
        
        float cosTheta = max(0.0f, L.dot(N));// * (L.dot(N));
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

struct Renderer {
    float screenW, screenH;
    float viewAngle;
    float distToScreen;
    
    vector<Shape* > shapesInScene;
    
    float ambientLightIntensity;
    vector<Light> lightsInScene;
    
    Vec3 camPosition;
    
    Renderer(float angle, float w, float h) {
        screen(w, h);
        
        screenW = w;
        screenH = h;
        viewAngle = angle;
        distToScreen = (w / 2 ) / tan(degToRadians(angle / 2));
        
        cout << "Dist to screen: " << distToScreen << endl;
        
        camPosition = Vec3(0, 0, 0);
    }
    
    void addLight(Light light) {
        lightsInScene.push_back(light);
    }
    
    Ray calculateRayForPixelOnScreen(int x, int y) {
        Vec3 pixelPos(x - w / 2, y - h / 2, distToScreen);
        
        return Ray(camPosition, pixelPos + camPosition);
    }
    
    Color calculateOnlySpecularHightlights(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = Vec3(0, 0, 0);
        
        for(Light& light : lightsInScene) {
            result = result + light.calculateSpecular(shape->material, pointOnObj, camPosition, objNormal);
        }
        
        return result;
    }
    
    Color calculateLighting(const Shape* shape, Vec3& objNormal, Vec3& pointOnObj) {
        Color result = shape->color * ambientLightIntensity;
        
        for(Light& light : lightsInScene) {
            Ray lightRay(pointOnObj, light.pos);
            Intersection closestIntersection;
            
            if(!findNearestRayIntersection(lightRay, shape, closestIntersection))
                result = result + light.evaluatePhongReflectionModel(shape->material, shape->color, objNormal, pointOnObj, camPosition);
        }
        
        return result.maxValue(1.0);
    }
    
    bool findNearestRayIntersection(const Ray& ray, const Shape* lastReflection, Intersection& closestIntersection) {
        Intersection intersections[10];
        bool hitAtLeastOneObject = false;
        float minDist = 100000000;
        
        closestIntersection.shape = NULL;
        
        int count = 0;
        
        for(Shape* shape : shapesInScene) {
            if(shape == lastReflection)
                continue;
            
            int totalIntersections = shape->calculateRayIntersections(ray, intersections);
            count += totalIntersections;
            
            findClosestIntersection(ray.v[0], closestIntersection, intersections, totalIntersections, minDist);
        }
        
        return closestIntersection.shape != NULL;
    }
    
    Color traceRay(const Ray& ray, int depth, const Shape* lastReflection) {
        Color rayColor = Vec3(0, 0, 0);
        Intersection closestIntersection;
        
        
        bool hitAtLeastOneObject = findNearestRayIntersection(ray, lastReflection, closestIntersection);
        
        if(hitAtLeastOneObject) {
            if(closestIntersection.shape->material.reflective) {
                if(depth < 5) {
                    Ray reflectedRay = ray.reflectAboutNormal(closestIntersection.normal, closestIntersection.pos);
                    
                    Color reflectedRayColor = traceRay(reflectedRay, depth + 1, closestIntersection.shape);
                    
                    float dot = -ray.dir.dot(closestIntersection.normal);
                    float alpha = .9;
                    float fresnelEffect = pow(1 - dot, 3) * (1 - alpha) + 1 * alpha;
                    
                    rayColor = (reflectedRayColor * fresnelEffect + calculateOnlySpecularHightlights(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos)).maxValue(1.0);
                }
            }
            else {
                rayColor = calculateLighting(closestIntersection.shape, closestIntersection.normal, closestIntersection.pos);
            }
        }
        
        return rayColor;
    }
    
    float distanceBetween(Vec3& v0, Vec3& v1) {
        return (v1 - v0).length();
    }
    
    void findClosestIntersection(Vec3 start, Intersection& closestIntersection, Intersection* intersections, int totalIntersections, float& minDist) {
        for(int i = 0; i < totalIntersections; ++i) {
            float dist = distanceBetween(start, intersections[i].pos);
            
            if(dist < minDist) {
                minDist = dist;
                closestIntersection = intersections[i];
            }
        }
    }
    
    void addObjectToScene(Shape* shape) {
        shapesInScene.push_back(shape);
    }
    
    void addTriangle(vector<Triangle> triangles) {
        for(Triangle t : triangles) {
            addObjectToScene(new Triangle(t));
        }
    }
    
    void addQuad(Vec3 v[4], Material& mat, Color color) {
        Triangle* t1 = new Triangle(v[0], v[2], v[0]);
        t1->material = mat;
        t1->color = color;
        
        Triangle* t2 = new Triangle(v[2], v[0], v[3]);
        t2->material = mat;
        t2->color = color;
        
        addObjectToScene(t1);
        addObjectToScene(t2);
    }
    
    void raytrace() {
        for(int i = 0; i < screenH; ++i) {
            for(int j = 0; j < screenW; ++j) {
                Ray ray = calculateRayForPixelOnScreen(j, i);
                
                Color rayColor = traceRay(ray, 0, NULL) * 255;
                
                pset(j, i, ColorRGB(rayColor.x, rayColor.y, rayColor.z));
            }
           
            redraw();
            redraw();
            cout << (i * 100 / screenH) << "%" << endl;
        }
    }
};





Sphere* sp;

struct ModelLoader {
    struct LineArgument {
        vector<string> part;
    };
    
    struct Line {
        string type;
        vector<LineArgument> arguments;
    };
    
    vector<Vec3> vertices;
    vector<Triangle> triangles;
    vector<Vec3> normals;
    
    vector<Triangle> loadFile(string fileName) {
        FILE* file = fopen(fileName.c_str(), "rb");
        if(!file)
            throw "Failed to load file: " + fileName;
         
        string fileContents;
        int c;
        while((c = fgetc(file)) != EOF) {
            fileContents += (char)c;
        }
        
        fileContents += '\0';
        fclose(file);
        
        vector<Line> lines = parseFile(fileContents);
        processLines(lines);
        
        return triangles;
    }
    
    void processLines(vector<Line>& lines) {
        for(Line& line : lines) {
            processLine(line);
        }
    }
    
    void processLine(Line& line) {
        if(line.type == "#" || line.type == "" || line.type == "s" || line.type == "g" || line.type == "usemtl" || line.type == "mtllib" || line.type == "vt") return;
        if(processVertex(line)) return;
        if(processFace(line)) return;
        if(processNormal(line)) return;
        
        throw "Unknown line type: " + line.type;
    }
    
    bool processVertex(Line& line) {
        if(line.type != "v")
            return false;
        
        assert(line.arguments.size() == 3);
        
        Vec3 v(
            atof(line.arguments[0].part[0].c_str()),
            -atof(line.arguments[1].part[0].c_str()) - 50,
            atof(line.arguments[2].part[0].c_str()) + 400
        );
        
        vertices.push_back(v);
        
        return true;
    }
    
    void addTriangle(int v0, int v1, int v2) {
        Material mat = Material(.9, 1.0, 50, true);
        
        Triangle triangle(vertices[v0], vertices[v1], vertices[v2]);
        triangle.color = Vec3(.5, .5, .5);
        triangle.material = mat;
        
        if(normals.size() >= vertices.size())
            triangle.setNormals(normals[v0], normals[v1], normals[v2]);
        
        triangles.push_back(triangle);
    }
    
    bool processFace(Line& line) {
        if(line.type != "f")
            return false;
        
        assert(line.arguments.size() == 3 || line.arguments.size() == 4);
        
        int v0 = atoi(line.arguments[0].part[0].c_str()) - 1;
        int v1 = atoi(line.arguments[1].part[0].c_str()) - 1;
        int v2 = atoi(line.arguments[2].part[0].c_str()) - 1;
        int v3 = 0;
        
        if(line.arguments.size() == 4)
            v3 = atoi(line.arguments[3].part[0].c_str()) - 1;
        
        addTriangle(v0, v1, v2);
        
        if(line.arguments.size() == 4) {
            addTriangle(v2, v3, v0);
        }
        
        return true;
    }
    
    bool processNormal(Line& line) {
        if(line.type != "vn")
            return false;
        
        assert(line.arguments.size() == 3);
        
        Vec3 n(
            atof(line.arguments[0].part[0].c_str()),
            -atof(line.arguments[1].part[0].c_str()),
            atof(line.arguments[2].part[0].c_str())
        );
        
        normals.push_back(n);
        
        return true;
    }
    
    char* findLineEnd(char* start) {
        while(*start && *start != '\n')
            ++start;
        
        return start;
    }
    
    char* consumeWhitespace(char* start, char* end) {
        while(start < end && (*start == ' ' || *start == '\t'))
            ++start;
        
        return start;
    }
    
    vector<Line> parseFile(string fileContents) {
        char* start = &fileContents[0];
        char* end;
        char* fileEnd = &fileContents[fileContents.size() - 1];
        vector<Line> lines;
        
        while(start < fileEnd) {
            end = findLineEnd(start);
            cout << "Line: " << string(start, end) << endl;
            lines.push_back(parseLine(start, end));
            start = end + 1;
        }
        
        for(Line line : lines) {
            cout << line.type << endl;
            
            for(LineArgument arg : line.arguments) {
                cout << "\t";
                
                for(int i = 0; i < arg.part.size(); ++i) {
                    cout << arg.part[i] << " ";
                }
                
                cout << endl;
            }
        }
        
        return lines;
    }
    
    Line parseLine(char* start, char* end) {
        char* startSave = start;
        
        start = consumeWhitespace(start, end);
        Line line;
        
        while(start < end && (*start == '#' || isalpha(*start))) {
            line.type += *start;
            ++start;
        }
        
        if(line.type == "#")
            return line;
        
        int count = 0;
        
        while(start < end) {
            start = consumeWhitespace(start, end);
            LineArgument lineArg;
            
            if(++count == 10000)
                throw "Too many iterations";
            
            while(start < end) {
                string arg;
                
                while(start < end && (*start == '.' || isdigit(*start) || *start == '-' || isalpha(*start) || *start == '_')) {
                    arg += *start;
                    ++start;
                }
                
                lineArg.part.push_back(arg);
                
                if(lineArg.part.size() > 10)
                    throw "Too many arguments for line: " + string(startSave, end);
                
                if(*start != '/')
                    break;
                
                ++start;
            }
            
            line.arguments.push_back(lineArg);
        }
        
        return line;
    }
};

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
        vector<Triangle> triangles = loader.loadFile("teapot.obj");
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
    Renderer renderer(60.0, 640, 480);
    
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


