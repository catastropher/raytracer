#pragma once

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
    #define CUDA_DEVICE __device__
#else
    #define CUDA_CALLABLE
    #define CUDA_DEVICE
#endif
