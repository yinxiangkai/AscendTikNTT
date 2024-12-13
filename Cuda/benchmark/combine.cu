#include "common.cuh"
#include "merge_ntt.cuh"
#include "parameters.cuh"
#include "tensor_ntt.cuh"
#include <asm-generic/errno.h>
#include <cstdint>
#include <iostream>
#include <vector>

// int main(int argc, char* argv[])
// {
//     CudaDevice();
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     int log_nttSize = argc > 1 ? atoi(argv[1]) : 16;
//     int batchSize0 = argc > 2 ? atoi(argv[2]) : 256;
//     int batchSize1 = argc > 3 ? atoi(argv[3]) : 0;
//     int batchSize = batchSize0 + batchSize1;
//     int nttSize = 1 << log_nttSize;
//     TensorNTTParameters tensorNTT(log_nttSize, 3);
//     NTTParameters mergeNTT(log_nttSize);

//     std::cout << "log_nttSize: " << log_nttSize << std::endl;
//     std::cout << "batchSize: " << batchSize << std::endl;
//     std::cout << "MBatchSize: " << batchSize0 << std::endl;
//     std::cout << "TBatchSize: " << batchSize1 << std::endl;

//     std::vector<uint64_t> h_input0(batchSize0 * mergeNTT.nttSize, 1);
//     std::vector<uint64_t> h_input1(batchSize1 * tensorNTT.nttSize, 1);

//     uint64_t* d_input0;
//     uint64_t* d_input1;

//     uint8_t* d_majorMatrix;
//     uint8_t* d_minorMatrix;
//     uint64_t* d_factorTable;
//     uint64_t* d_unityRootReverseTable;
//     uint64_t* d_inverseRootUnityReverseTable;

//     CUDA_CHECK(cudaMalloc((void**)&d_input0, h_input0.size() * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc((void**)&d_input1, h_input1.size() * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_unityRootReverseTable, mergeNTT.rootSize * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_inverseRootUnityReverseTable, mergeNTT.rootSize * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc((void**)&d_majorMatrix, tensorNTT.majorMatrix.size() * sizeof(uint8_t)));
//     CUDA_CHECK(cudaMalloc((void**)&d_minorMatrix, tensorNTT.minorMatrix.size() * sizeof(uint8_t)));
//     CUDA_CHECK(cudaMalloc((void**)&d_factorTable, tensorNTT.factorTable.size() * sizeof(uint64_t)));

//     CUDA_CHECK(cudaMemcpy(d_input0, h_input0.data(), h_input0.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_input1, h_input1.data(), h_input1.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_unityRootReverseTable, mergeNTT.unityRootReverseTable.data(), mergeNTT.rootSize * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_inverseRootUnityReverseTable, mergeNTT.inverseUnityRootReverseTable.data(), mergeNTT.rootSize * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_majorMatrix, tensorNTT.majorMatrix.data(), tensorNTT.majorMatrix.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_minorMatrix, tensorNTT.minorMatrix.data(), tensorNTT.minorMatrix.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_factorTable, tensorNTT.factorTable.data(), tensorNTT.factorTable.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

//     cudaStream_t stream0;
//     cudaStream_t stream1;
//     cudaStreamCreate(&stream0);
//     cudaStreamCreate(&stream1);
//     float runTime = 0;
//     if ((batchSize0 != 0) && (batchSize1 != 0))
//     {
//         GPU_TensorNTT_3(d_input1, d_input1, d_majorMatrix, d_minorMatrix, d_factorTable, tensorNTT.log_nttSize, tensorNTT.modulus, batchSize1, tensorNTT.nttSize, stream1);
//         GPU_NTT(d_input0, d_input0, d_unityRootReverseTable, log_nttSize, mergeNTT.modulus, batchSize0, stream0);
//         CUDA_CHECK(cudaGetLastError());
//         CUDA_CHECK(cudaEventRecord(start));
//         for (int i = 0; i < 1000; i++)
//         {

//             GPU_NTT(d_input0, d_input0, d_unityRootReverseTable, log_nttSize, mergeNTT.modulus, batchSize0, stream0);
//             GPU_TensorNTT_3(d_input1, d_input1, d_majorMatrix, d_minorMatrix, d_factorTable, tensorNTT.log_nttSize, tensorNTT.modulus, batchSize1, tensorNTT.nttSize, stream1);
//         }
//         CUDA_CHECK(cudaEventRecord(stop));
//         CUDA_CHECK(cudaEventSynchronize(stop));
//         float milliseconds = 0;
//         CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
//         //  runTime += milliseconds;
//         std::cout << "time: " << milliseconds << " us" << std::endl;
//     }
//     else if (batchSize0 != 0)
//     {
//         for (int i = 0; i < 1000; i++)
//         {
//             GPU_NTT(d_input0, d_input0, d_unityRootReverseTable, log_nttSize, mergeNTT.modulus, batchSize0, stream0);
//             CUDA_CHECK(cudaGetLastError());
//             CUDA_CHECK(cudaEventRecord(start));
//             GPU_NTT(d_input0, d_input0, d_unityRootReverseTable, log_nttSize, mergeNTT.modulus, batchSize0, stream0);
//             CUDA_CHECK(cudaEventRecord(stop));
//             CUDA_CHECK(cudaEventSynchronize(stop));
//             float milliseconds = 0;
//             CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
//             runTime += milliseconds;
//         }
//         std::cout << "time: " << runTime << " us" << std::endl;
//     }
//     else if (batchSize1 != 0)
//     {
//         for (int i = 0; i < 1000; i++)
//         {
//             GPU_TensorNTT_3(d_input1, d_input1, d_majorMatrix, d_minorMatrix, d_factorTable, tensorNTT.log_nttSize, tensorNTT.modulus, batchSize1, tensorNTT.nttSize, stream1);
//             CUDA_CHECK(cudaGetLastError());
//             CUDA_CHECK(cudaEventRecord(start));
//             GPU_TensorNTT_3(d_input1, d_input1, d_majorMatrix, d_minorMatrix, d_factorTable, tensorNTT.log_nttSize, tensorNTT.modulus, batchSize1, tensorNTT.nttSize, stream1);
//             CUDA_CHECK(cudaEventRecord(stop));
//             CUDA_CHECK(cudaEventSynchronize(stop));
//             float milliseconds = 0;
//             CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
//             runTime += milliseconds;
//         }
//         std::cout << "time: " << runTime << " us" << std::endl;
//     }
//     return 0;
// }


int main(int argc, char* argv[])
{
    CudaDevice();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int log_nttSize = argc > 1 ? atoi(argv[1]) : 16;
    int batchSize = argc > 2 ? atoi(argv[2]) : 16;
    TensorNTTParameters tensorNTT(log_nttSize, 3);
    NTTParameters mergeNTT(log_nttSize);

    std::cout << "log_nttSize: " << log_nttSize << std::endl;
    std::cout << "batchSize: " << batchSize << std::endl;

    std::vector<uint64_t> h_input0(mergeNTT.nttSize, 1);

    uint64_t* d_input0;
    uint64_t* d_input1;

    uint8_t* d_majorMatrix;
    uint8_t* d_minorMatrix;
    uint64_t* d_factorTable;
    uint64_t* d_unityRootReverseTable;
    uint64_t* d_inverseRootUnityReverseTable;

    CUDA_CHECK(cudaMalloc((void**)&d_input0, h_input0.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_input1, h_input0.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_unityRootReverseTable, mergeNTT.rootSize * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_inverseRootUnityReverseTable, mergeNTT.rootSize * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_majorMatrix, tensorNTT.majorMatrix.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_minorMatrix, tensorNTT.minorMatrix.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_factorTable, tensorNTT.factorTable.size() * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_input0, h_input0.data(), h_input0.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input1, h_input0.data(), h_input0.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_unityRootReverseTable, mergeNTT.unityRootReverseTable.data(), mergeNTT.rootSize * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverseRootUnityReverseTable, mergeNTT.inverseUnityRootReverseTable.data(), mergeNTT.rootSize * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_majorMatrix, tensorNTT.majorMatrix.data(), tensorNTT.majorMatrix.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_minorMatrix, tensorNTT.minorMatrix.data(), tensorNTT.minorMatrix.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_factorTable, tensorNTT.factorTable.data(), tensorNTT.factorTable.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    std::vector<cudaStream_t> streams(batchSize);
    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    float runTime = 0;
    for (int i = 0; i < 1000; i++)
    {
        #pragma unroll
        for (int j= 0; j < 1000; j++)
        {
            GPU_INTT(d_input1, d_input1, d_unityRootReverseTable, log_nttSize, mergeNTT.nttInv, mergeNTT.modulus, 1);
        }
        CUDA_CHECK(cudaEventRecord(start));
        GPU_NTT(d_input0, d_input0, d_unityRootReverseTable, log_nttSize, mergeNTT.modulus, 1, stream1);
        // GPU_TensorNTT_3(d_input0, d_input0, d_majorMatrix, d_minorMatrix, d_factorTable, tensorNTT.log_nttSize, tensorNTT.modulus, 1, tensorNTT.nttSize, stream1);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        runTime += milliseconds;
    }
    std::cout << "time: " << runTime << " us" << std::endl;

    return 0;
}
