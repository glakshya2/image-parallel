cmake_minimum_required(VERSION 3.30.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(image-parallel LANGUAGES CXX CUDA)

include_directories("${PROJECT_SOURCE_DIRECTORY}/include")

file(GLOB_RECURSE SOURCES src/*)

add_executable(${PROJECT_NAME} main.cpp ${SOURCES})