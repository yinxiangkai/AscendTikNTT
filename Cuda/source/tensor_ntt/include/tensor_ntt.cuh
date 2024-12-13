#pragma once
#include "modular.cuh"
#include <common.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>
#include <sys/types.h>
using namespace nvcuda;

// 方案0:不递归
__global__ void tensorCore_0(const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus, int nttSize);
__global__ void tensorCore_0(const uint64_t* input, uint64_t* output, const uint8_t* minorMatrix, Modulus modulus, int nttSize);
__host__ void GPU_TensorNTT_0(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize);
// 方案1 仅递归
__global__ void tensorCore_1(const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus, int nttSize);
__global__ void tensorCore_1(int minorSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, const uint8_t* minorMatrix, Modulus modulus,
                             int nttSize);
__host__ void GPU_TensorNTT_1(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize);

// 方案2 数据读取优化
__global__ void tensorCore_2(int flag, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus, int nttSize);
__global__ void tensorCore_2(const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus, int nttSize);
__global__ void tensorCore_2(int minorSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, const uint8_t* minorMatrix, Modulus modulus,
                             int nttSize);
__host__ void GPU_TensorNTT_2(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize, cudaStream_t stream);

// 方案3 数据传输优化 16 16*16*16 非连续
__global__ void tensorCore_3A(int logStep, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus);
__global__ void tensorCore_3B(int log_minorSize, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              const uint8_t* minorMatrix, Modulus modulus);
__host__ void GPU_TensorNTT_3(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                         int batchSize, int nttSize, cudaStream_t stream);
// 方案4 16 16*16*16 连续
__global__ void __launch_bounds__(512)
    tensorCore_4A(int logStep, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus);
__global__ void tensorCore_4B(int log_minorSize, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              const uint8_t* minorMatrix, Modulus modulus);
__host__ void GPU_TensorNTT_4(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize, cudaStream_t stream);

// 方案5 32 16*16*16 非连续
__global__ void tensorCore_5A(int logStep, int log_nttSize, int log_subNum, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              Modulus modulus);
__global__ void tensorCore_5B(int log_minorSize, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              const uint8_t* minorMatrix, Modulus modulus);
__host__ void GPU_TensorNTT_5(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize);

// 方案6 32 16*16*16 连续
__global__ void __launch_bounds__(1024)
    tensorCore_6A(int logStep, int log_nttSize, int log_subNum, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus);
__global__ void tensorCore_6B(int log_minorSize, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              const uint8_t* minorMatrix, Modulus modulus);
__host__   void GPU_TensorNTT_6(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize);

// 方案7 32 32*8*16 非连续
__global__ void tensorCore_7A(int logStep, int log_nttSize, int log_subNum, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus);
__global__ void tensorCore_7B(int log_minorSize, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              const uint8_t* minorMatrix, Modulus modulus);
__host__   void GPU_TensorNTT_7(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize);

// 方案8 32 32*8*16 连续
__global__ void __launch_bounds__(1024) tensorCore_8(const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                                                        Modulus modulus, int nttSize, int logStep);
__global__ void __launch_bounds__(1024) tensorCore_8(int log_minorSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                                                     const uint8_t* minorMatrix, Modulus modulus, int nttSize);
__host__  void GPU_TensorNTT_8(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                         int batchSize, int nttSize);