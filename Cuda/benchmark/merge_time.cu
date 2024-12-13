// 本文件用于测试merge_ntt。
#include "common.cuh"
#include "merge_ntt.cuh"
#include "parameters.cuh"
#include <cstdint>
#include <iostream>

int main(int argc, char* argv[])
{
    CudaDevice();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int log_nttSize = argc > 1 ? atoi(argv[1]) : 12;
    int batchSize = argc > 2 ? atoi(argv[2]) : 1;
    std::cout << "batchSize: " << batchSize << std::endl;
    std::cout << "log_nttSize: " << log_nttSize << std::endl;
    NTTParameters parameters(log_nttSize);

    std::vector<uint64_t> input(batchSize * parameters.nttSize, 1);

    uint64_t *d_input, *d_output, *d_unityRootReverseTable, *d_inverseRootUnityReverseTable;
    Modulus* d_modulus;

    cudaMalloc(&d_input, batchSize * parameters.nttSize * sizeof(uint64_t));
    cudaMalloc(&d_output, batchSize * parameters.nttSize * sizeof(uint64_t));
    cudaMalloc(&d_unityRootReverseTable, parameters.rootSize * sizeof(uint64_t));
    cudaMalloc(&d_inverseRootUnityReverseTable, parameters.rootSize * sizeof(uint64_t));
    cudaMalloc(&d_modulus, sizeof(Modulus));

    cudaMemcpy(d_input, input.data(), batchSize * parameters.nttSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_unityRootReverseTable, parameters.unityRootReverseTable.data(), parameters.rootSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inverseRootUnityReverseTable, parameters.inverseUnityRootReverseTable.data(), parameters.rootSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float runTime;

    for (int i = 0; i < 1000; i++)
    {
        GPU_NTT(d_input, d_output, d_unityRootReverseTable, log_nttSize, parameters.modulus, batchSize, stream);
        cudaEventRecord(start);
        GPU_NTT(d_input, d_output, d_unityRootReverseTable, log_nttSize, parameters.modulus, batchSize, stream);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        runTime += elapsedTime;
    }

    std::cout << "GPU_NTT time: " << runTime << " us" << std::endl;

    return 0;
}
