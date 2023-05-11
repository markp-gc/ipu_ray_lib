cmake_minimum_required(VERSION 3.16)

project(neural_networks)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-sign-compare -Wno-error=deprecated-declarations)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(
  ${CMAKE_SOURCE_DIR}/include
)

file(GLOB NN_HEADERS ${PROJECT_SOURCE_DIR}/*.hpp)
file(GLOB NN_SRC ${PROJECT_SOURCE_DIR}/*.cpp)
add_library(neural_networks ${NN_HEADERS} ${NN_SRC})
