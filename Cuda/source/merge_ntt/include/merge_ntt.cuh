#pragma once
#include "modular.cuh"
#include <cstdint>

__device__ void CooleyTukeyUnit(uint64_t& U, uint64_t& V, uint64_t& omega, Modulus& modulus);

__device__ void GentlemanSandeUnit(uint64_t& U, uint64_t& V, uint64_t& omega, Modulus& modulus);


__global__ void NTTCore(uint64_t* input, uint64_t* output, uint64_t* rootUnityTable, Modulus modulus, int sharedIndex, int logm, int roundCount, int log_nttSize,
                        cudaStream_t stream);


__global__ void INTTCore(uint64_t* input, uint64_t* output, uint64_t* inverse_rootUnityTable, Modulus modulus,
                         int sharedIndex, int logm, int k, int roundCount, int log_nttSize, uint64_t n_inverse,
                         bool is_LastKernel);

__host__ void GPU_NTT(uint64_t* input, uint64_t* output, uint64_t* rootUnityTable, int log_nttSize, Modulus modulus, int batchSize, cudaStream_t stream);

__host__ void GPU_INTT(uint64_t* input, uint64_t* output, uint64_t* rootUnityTable, int log_nttSize, uint64_t nttInv, Modulus modulus, int batchSize);