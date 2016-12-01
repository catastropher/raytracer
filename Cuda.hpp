#pragma once

#include "config.hpp"
#include "Scene.hpp"
#include "Renderer.hpp"

void launchCudaKernel(float angle, int w, int h, Scene scene, Renderer& hostRenderer);

