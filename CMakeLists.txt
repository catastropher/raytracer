cmake_minimum_required(VERSION 2.6)

set(RAYTRACE_SOURCES raytrace.cpp)

project(raytrace)

find_package(SDL)
find_package(CUDA)

if(SDL_FOUND)
  list(APPEND RAYTRACE_SOURCES quickcg.cpp)
  add_definitions(-D__WITH_SDL__)
endif()

set(USE_CUDA false)

if(CUDA_FOUND)
    set(USE_CUDA true)
    add_definitions(-D__WITH_CUDA__)
    list(APPEND RAYTRACE_SOURCES Cuda.cpp)
endif()

if(USE_CUDA)
    cuda_add_executable(raytrace
        ${RAYTRACE_SOURCES}
    )
else()
    add_executable(raytrace
        ${RAYTRACE_SOURCES}
    )
endif()
  
if(SDL_FOUND)
  target_link_libraries(raytrace SDL)
endif()

set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++11")
