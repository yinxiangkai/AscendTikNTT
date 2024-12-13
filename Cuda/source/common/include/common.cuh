#pragma once
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <string>



/*
 * 检测cuda执行情况
 */
#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        const cudaError_t error_code = call;                                           \
        if (error_code != cudaSuccess) {                                               \
            std::cout << "\033[0;31m"                                                  \
                      << "CUDA Error" << std::endl;                                    \
            std::cout << "\033[0;31m"                                                  \
                      << "File:" << __FILE__ << std::endl;                             \
            std::cout << "\033[0;31m"                                                  \
                      << "Line:" << __LINE__ << std::endl;                             \
            std::cout << "\033[0;31m"                                                  \
                      << "Error code:" << error_code << std::endl;                     \
            std::cout << "\033[0;31m"                                                  \
                      << "Error text:" << cudaGetErrorString(error_code) << std::endl; \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define CUDA_EXEC(call)                                                                \
    {                                                                                  \
        call;                                                                          \
        cudaError_t error_code = cudaGetLastError();                                   \
        if (error_code != cudaSuccess) {                                               \
            std::cout << "\033[0;31m"                                                  \
                      << "CUDA Error" << std::endl;                                    \
            std::cout << "\033[0;31m"                                                  \
                      << "File:" << __FILE__ << std::endl;                             \
            std::cout << "\033[0;31m"                                                  \
                      << "Line:" << __LINE__ << std::endl;                             \
            std::cout << "\033[0;31m"                                                  \
                      << "Error code:" << error_code << std::endl;                     \
            std::cout << "\033[0;31m"                                                  \
                      << "Error text:" << cudaGetErrorString(error_code) << std::endl; \
            exit(1);                                                                   \
        }                                                                              \
    }


/*
 * 功能函数
 */

void CudaDeviceInformation(int deviceId);
void customAssert(bool condition, const std::string& errorMessage);
void PassMessage(bool condition, const std::string& message);
void CudaDevice();
size_t getMaxSharedMem();

