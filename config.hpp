#pragma once

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
    #define CUDA_DEVICE __device__
#else
    #define CUDA_CALLABLE
    #define CUDA_DEVICE
#endif

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define THREADS_IN_BLOCK (BLOCK_WIDTH * BLOCK_HEIGHT)

static inline float degToRadians(float deg) {
    const float PI = 3.1415926535897;
    return deg * PI / 180.0;
}

template<typename T>
CUDA_CALLABLE T minimum(const T& a, const T& b) {
    return (a < b ? a : b);
}
