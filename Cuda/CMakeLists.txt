cmake_minimum_required(VERSION 3.12)
project(TensorNTT VERSION 1.0 LANGUAGES C CXX CUDA ASM)

# 设置CMake的策略
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)  
endif()


set(CMAKE_CUDA_ARCHITECTURES "90")
# 设置编译选项
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置编译器
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CUDA_COMPILER nvcc)

option(NVTX "Enable NVTX profiling" OFF)
option(DEBUG "Enable Debug mode" OFF)

# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -arch=sm_90 -O3")  
# H100、4090、A100
set(CMAKE_CUDA_ARCHITECTURES "90,89,80")

# 设置CMake的策略
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)  # 使用新的CMake策略，允许设置CMAKE_CUDA_ARCHITECTURES
endif()

if(DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -arch=sm_90 -g -G")  
    message(STATUS "DEBUG MODE")
else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -arch=sm_90 -O3")  
    message(STATUS "Release MODE")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if(NVTX)
	add_definitions(-DNVTX)
    message(STATUS "NVTX MODE")
ENDIF()

add_subdirectory(source)
add_subdirectory(benchmark)