cmake_minimum_required(VERSION 2.6)

project(raytrace)

add_executable(raytrace
  raytrace.cpp
  quickcg.cpp
)

find_package(SDL REQUIRED)

target_link_libraries(raytrace SDL)

set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++11")