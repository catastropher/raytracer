#pragma once

#include <vector>
#include <string>
#include <cassert>

#include "Vec3.hpp"
#include "Material.hpp"
#include "Triangle.hpp"

struct ModelLoader {
    struct LineArgument {
        std::vector<std::string> part;
    };
    
    struct Line {
        std::string type;
        std::vector<LineArgument> arguments;
    };
    
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    std::vector<Vec3> normals;
    
    std::vector<Triangle> loadFile(std::string fileName) {
        FILE* file = fopen(fileName.c_str(), "rb");
        if(!file)
            throw "Failed to load file: " + fileName;
         
        std::string fileContents;
        int c;
        while((c = fgetc(file)) != EOF) {
            fileContents += (char)c;
        }
        
        fileContents += '\0';
        fclose(file);
        
        std::vector<Line> lines = parseFile(fileContents);
        processLines(lines);
        
        return triangles;
    }
    
    void processLines(std::vector<Line>& lines) {
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
    
    std::vector<Line> parseFile(std::string fileContents) {
        char* start = &fileContents[0];
        char* end;
        char* fileEnd = &fileContents[fileContents.size() - 1];
        std::vector<Line> lines;
        
        while(start < fileEnd) {
            end = findLineEnd(start);
            std::cout << "Line: " << std::string(start, end) << std::endl;
            lines.push_back(parseLine(start, end));
            start = end + 1;
        }
        
        for(Line line : lines) {
            std::cout << line.type << std::endl;
            
            for(LineArgument arg : line.arguments) {
                std::cout << "\t";
                
                for(int i = 0; i < arg.part.size(); ++i) {
                    std::cout << arg.part[i] << " ";
                }
                
                std::cout << std::endl;
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
                std::string arg;
                
                while(start < end && (*start == '.' || isdigit(*start) || *start == '-' || isalpha(*start) || *start == '_')) {
                    arg += *start;
                    ++start;
                }
                
                lineArg.part.push_back(arg);
                
                if(lineArg.part.size() > 10)
                    throw "Too many arguments for line: " + std::string(startSave, end);
                
                if(*start != '/')
                    break;
                
                ++start;
            }
            
            line.arguments.push_back(lineArg);
        }
        
        return line;
    }
};

