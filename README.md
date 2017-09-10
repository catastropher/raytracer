# Raytracer
This is a raytracer written entirely from scratch in C++. It can run on both the CPU and a CUDA card if one is present (this project was written and tested on a GeFroce GTX Titan). It was written for a graudate level parallel processing course in December 2016. In one scene with 100,000 triangles, it took 66,817,810,944 triangle-ray intersections (which humerously caused integer overflow at first). Rendering this on the CPU takes ~26 minutes. By effectively using the CUDA hardware, this time was reduced to 1.3 seconds.

### Sample Output Image
![image](https://user-images.githubusercontent.com/5026862/30245431-5261b930-95a7-11e7-8b18-4614d56bb6f9.png)

### Features
* Phong Shading
* Shadows
* Mirrors/glass
* Rendering of spheres, quads, triangles, and models
* Loading of Wavefront .obj files for models (but they can only be made of triangles and quads for now)
* Can be run on the CPU or a CUDA card
* Specifically optimized to take advantage of CUDA (see below for experimental results)

### Dependicies
* SDL 1.2
* CMake
* CUDA (if building for a CUDA enabled card)

### Build Instructions
#### Without CUDA
```
mkdir build
cd build
cmake ..
make
./raytrace
```

#### With CUDA
```
mkdir build
cd build
cmake .. -DUSE_CUDA
make
./raytrace
```

### Command line options
* -w: sets the width of the output image
* -h: sets the height of the output image
* --save: sets the name of the output image
* --load: loads in an output image for viewing

### Running on a remote server that has a CUDA card
Modify the run script and replace my account name with your computer account name. You may also want to change the location of where to build the project remotely (see remote-run). Then, run the "run" script, which will run it remotely, copy back the resulting image, and display it localy.
