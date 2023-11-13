cmake_minimum_required(VERSION 3.5)
project(mat_multi)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_BUILD_TYPE "Release" )

set(CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=sm_86;-g;-O2;-w")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.1)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)

#cublas
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

cuda_add_library(my_cu SHARED
                       ${PROJECT_SOURCE_DIR}/cuda_kernel/gemm_naive.cu
                       ${PROJECT_SOURCE_DIR}/cuda_kernel/gemm_tiling.cu 
                       ${PROJECT_SOURCE_DIR}/cuda_kernel/gemm_tiling_wpt.cu
                       ${PROJECT_SOURCE_DIR}/cuda_kernel/gemm_tiling_wpt_block.cu
                       ${PROJECT_SOURCE_DIR}/cuda_kernel/gemm_tiling_wpt_block_rect.cu)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

include_directories(${CUDA_INCLUDE_DIRS} ${COMMON_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(mat_multi ${PROJECT_SOURCE_DIR}/main.cpp )
target_link_libraries(mat_multi  ${CUDA_LIBRARIES} my_cu cublas)

add_definitions(-O2 -pthread)