project(embree_test)
cmake_minimum_required(VERSION 3.16)

# Use C++20 to guarantee std::vector alignment for Embree:
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(embree_DIR /usr/include)

# You can use this to enable or disable double precision fallback
# in the triangle test. By default it is disabled as IPU has no
# hardware double support (doubles will be emulated):
set(ALLOW_DOUBLE_FALLBACK OFF CACHE BOOL "Enable double precision fallback (emulated on IPU).")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
#find_package(embree 3.0 REQUIRED)
find_package(assimp 5 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

execute_process(COMMAND bash "-c" "popc --version | cut -d ' ' -f3 | head -1" OUTPUT_VARIABLE POPLAR_VERSION)
string(REPLACE "." ";" VERSION_LIST ${POPLAR_VERSION})
list(GET VERSION_LIST 0 POPLAR_VERSION_MAJOR)
list(GET VERSION_LIST 1 POPLAR_VERSION_MINOR)
list(GET VERSION_LIST 2 POPLAR_VERSION_PATCH)
message(STATUS "Detected Poplar version ${POPLAR_VERSION_MAJOR}.${POPLAR_VERSION_MINOR}.${POPLAR_VERSION_PATCH}")

message(STATUS "HDF5 LIBRARIES: ${HDF5_LIBRARIES}")
add_definitions(${HDF5_DEFINITIONS})

if(ALLOW_DOUBLE_FALLBACK)
  set(USE_DOUBLE "1")
else()
  set(USE_DOUBLE "0")
endif()
message(STATUS "Setting ALLOW_DOUBLE_FALLBACK: " ${USE_DOUBLE})

add_compile_definitions(ALLOW_DOUBLE_FALLBACK=${USE_DOUBLE})

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ASSIMP_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/ext
  ${HDF5_INCLUDE_DIRS}
)

link_directories(${ASSIMP_LIBRARY_DIRS})

# Add the neural network library:
add_subdirectory(${CMAKE_SOURCE_DIR}/src/keras)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/neural_networks)

# Add IPU ray tracing library:
FILE(GLOB_RECURSE IPU_RAYLIB_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp)
FILE(GLOB_RECURSE IPU_RAYLIB_SRC ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/ext/math/*.cpp)
add_library(ipu_ray_lib SHARED ${IPU_RAYLIB_SRC})
target_include_directories(ipu_ray_lib PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/embree_utils)

# Poplar lib defines:
set(POPLAR_LINK_LIBRARIES -lpoplin -lpopnn -lpoplar -lpopops -lpoputil -lpoprand -lgcl -lpvti)

# Add a trace executable:
add_executable(trace trace.cpp TraceCodelets.gp)
target_include_directories(trace PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/embree_utils)
target_link_libraries(trace
  ipu_ray_lib
  #embree
  embree3
  Boost::program_options
  ${OpenCV_LIBS}
  ${ASSIMP_LIBRARIES}
  OpenMP::OpenMP_CXX
  Eigen3::Eigen
  ${HDF5_LIBRARIES}
  ${POPLAR_LINK_LIBRARIES})

# Explicitly define the subset of the library source files needed to build IPU codelet:
set(CODELET_SRC
  ${CMAKE_SOURCE_DIR}/src/Mesh.cpp
  ${CMAKE_SOURCE_DIR}/src/CompactBVH2Node.cpp
  ${CMAKE_SOURCE_DIR}/src/Primitives.cpp
  ${CMAKE_SOURCE_DIR}/codelets/TraceCodelets.cpp
  ${CMAKE_SOURCE_DIR}/ext/math/sincos.cpp
)

# Custom build step for IPU codelets:
add_custom_command(
  PRE_BUILD
  MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/codelets/TraceCodelets.cpp
  DEPENDS ${IPU_RAYLIB_HEADERS} ${CODELET_SRC}
  COMMAND popc -DALLOW_DOUBLE_FALLBACK=${USE_DOUBLE} -O3 -Werror -Wdouble-promotion --target ipu2 -I${CMAKE_SOURCE_DIR}/codelets -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/ext -I${CMAKE_SOURCE_DIR}/ext/math ${CODELET_SRC} -o TraceCodelets.gp
  OUTPUT TraceCodelets.gp
  WORKING_DIRECTORY ${CMAKE_BUILD_DIR}
)

# Tests
add_executable(tests ${CMAKE_SOURCE_DIR}/tests/test.cpp)
target_link_libraries(tests ipu_ray_lib ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${POPLAR_LINK_LIBRARIES})
add_test(test1 tests)
