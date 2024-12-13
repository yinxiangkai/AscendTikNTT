#include "common.cuh"
#include <cstdio>
#include <iostream>


void CudaDeviceInformation(int deviceId)
{
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, deviceId);
    std::cout << "\033[1;34m" << "-------------------------------------------------------------------------------------"
              << "\033[0;37m" << std::endl;
    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "  Device " << deviceId << ": " << prop.name << std::endl;
    std::cout << "      Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "      Shared memory per block: " << prop.sharedMemPerBlock / 1024 << "KB" << std::endl;
    std::cout << "\033[1;34m" << "-------------------------------------------------------------------------------------"
              << "\033[0;37m" << std::endl;
}

void customAssert(bool condition, const std::string& errorMessage)
{
    if (!condition) {
        std::cerr << "\033[1;33m" << "Custom assertion failed: " << errorMessage << "\033[0;37m" << std::endl;
    }
}

void PassMessage(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "\033[1;31m" << "Failed: " << message << "\033[0;37m" << std::endl;
    }
    else {
        std::cout << "\033[1;32m" << "Passed: " << message << "\033[0;37m" << std::endl;
    }
}
void CudaDevice()
{
    cudaDeviceProp deviceProp;
    int deviceID = 0;

    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceID));
    printf("GPU Device %d: %s (compute capability %d.%d)\n", deviceID, deviceProp.name, deviceProp.major, deviceProp.minor);
}

size_t getMaxSharedMem()
{
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int maxSharedMemoryPerMultiprocessor;
    CUDA_CHECK(
        cudaDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    std::cout << "Max shared memory per block: " << maxSharedMemoryPerMultiprocessor - 1024 << std::endl;
    return maxSharedMemoryPerMultiprocessor - 1024;   // 1024 is the size of the stack
}

