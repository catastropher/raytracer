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
