#pragma once

#include "Vec3.hpp"

using Color = Vec3;

struct Material {
    float diffuse;
    float specular;
    float alpha;
    bool reflective;
    
    Material(float d, float s, float a, bool reflective_) : diffuse(d), specular(s), alpha(a), reflective(reflective_) { }
    Material() { }
};