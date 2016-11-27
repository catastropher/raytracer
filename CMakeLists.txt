cmake_minimum_required(VERSION 2.6)

set(RAYTRACE_SOURCES raytrace.cpp)

project(raytrace)

find_package(SDL)

if(SDL_FOUND)
  list(APPEND RAYTRACE_SOURCES quickcg.cpp)
  add_definitions(-D__WITH_SDL__)
endif()
  
add_executable(raytrace
  ${RAYTRACE_SOURCES}
)
  
if(SDL_FOUND)
  target_link_libraries(raytrace SDL)
endif()

set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++11")