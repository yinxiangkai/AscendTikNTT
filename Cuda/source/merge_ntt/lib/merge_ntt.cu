#include "common.cuh"
#include "merge_ntt.cuh"
#include <cstdint>
#include <iostream>

__device__ void CooleyTukeyUnit(uint64_t& U, uint64_t& V, uint64_t& omega, Modulus& modulus)
{
    uint64_t uTemp = U;
    uint64_t vTemp = Barrett64_gpu::mult(V, omega, modulus);
    U = Barrett64_gpu::add(uTemp, vTemp, modulus);
    V = Barrett64_gpu::sub(uTemp, vTemp, modulus);
}

__device__ void GentlemanSandeUnit(uint64_t& U, uint64_t& V, uint64_t& omega, Modulus& modulus)
{

    uint64_t uTemp = U;
    uint64_t vTemp = V;
    U = Barrett64_gpu::add(uTemp, vTemp, modulus);
    vTemp = Barrett64_gpu::sub(uTemp, vTemp, modulus);
    V = Barrett64_gpu::mult(vTemp, omega, modulus);
}

__global__ void NTTCore(uint64_t* input, uint64_t* output, uint64_t* rootUnityTable, Modulus modulus, int sharedIndex, int logm, int roundCount, int log_nttSize)
{
    // roundCount 循环轮次
    // log  block内的线程数的log2
    //  当前的线程信息
    const int idx_x = threadIdx.x;    // 第几个线程
    const int idx_y = threadIdx.y;    // 第几种蝶形变换
    const int block_x = blockIdx.x;   // 线程块编号
    const int block_y = blockIdx.y;   // 分组编号(共用一个omega的是一个组)
    const int block_z = blockIdx.z;   // NTT编号

    // 申请共享内存(512*8)=4KB
    extern __shared__ uint64_t shared_memory[];

    int t_2 = log_nttSize - logm - 1;
    uint32_t offset = 1 << (log_nttSize - logm - 1);   // 分组内蝶形变换的次数
    int t_ = sharedIndex;                              // 线程数的log2，主要是用来调整线程与数据的对应关系

    // uint32  // 第几个NTT的第几个分组中的第几个线程块中的第几种蝶形变换
    uint32_t global_addresss =
        idx_x + (uint32_t)(idx_y * (offset / (1 << (roundCount - 1)))) + (uint32_t)(blockDim.x * block_x) + (uint32_t)(2 * block_y * offset) + (uint32_t)(block_z << log_nttSize);

    uint32_t omega_addresss = idx_x + (uint32_t)(idx_y * (offset / (1 << (roundCount - 1)))) + (uint32_t)(blockDim.x * block_x) + (uint32_t)(block_y * offset);

    // 分配线程对应的共享内存，一个线程对应两个数据，即蝶形变换的两个数据。
    uint32_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // 每个线程读取两个数据到共享内存
    shared_memory[shared_addresss] = input[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] = input[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;   // 重新调整线程和数据的对应关系
    uint32_t current_root_index;

#pragma unroll
    for (int lp = 0; lp < roundCount; lp++)
    {
        __syncthreads();
        current_root_index = (omega_addresss >> t_2);
        CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t], rootUnityTable[current_root_index], modulus);
        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();
    uint64_t temp1 = 2;
    uint64_t temp = Barrett64_gpu::mult(temp1, temp1, modulus);
    output[global_addresss] = shared_memory[shared_addresss];
    output[global_addresss + offset] = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void INTTCore(uint64_t* input, uint64_t* output, uint64_t* inverse_rootUnityTable, Modulus modulus, int sharedIndex, int logm, int k, int roundCount, int log_nttSize,
                         uint64_t n_inverse, bool is_LastKernel)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    extern __shared__ uint64_t shared_memory[];

    int t_2 = log_nttSize - logm - 1;
    uint32_t offset = 1 << (log_nttSize - k - 1);
    int t_ = (sharedIndex + 1) - roundCount;
    int loops = roundCount;

    uint32_t global_addresss =
        idx_x + (uint32_t)(idx_y * (offset / (1 << (roundCount - 1)))) + (uint32_t)(blockDim.x * block_x) + (uint32_t)(2 * block_y * offset) + (uint32_t)(block_z << log_nttSize);

    uint32_t omega_addresss = idx_x + (uint32_t)(idx_y * (offset / (1 << (roundCount - 1)))) + (uint32_t)(blockDim.x * block_x) + (uint32_t)(block_y * offset);
    uint32_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = input[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] = input[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    uint32_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        current_root_index = (omega_addresss >> t_2);

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t], inverse_rootUnityTable[current_root_index], modulus);

        t = t << 1;
        t_2 += 1;
        t_ += 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (is_LastKernel)
    {

        output[global_addresss] = Barrett64_gpu::mult(shared_memory[shared_addresss], n_inverse, modulus);
        output[global_addresss + offset] = Barrett64_gpu::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)], n_inverse, modulus);
    }
    else
    {
        output[global_addresss] = shared_memory[shared_addresss];
        output[global_addresss + offset] = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__host__ void GPU_NTT(uint64_t* input, uint64_t* output, uint64_t* rootUnityTable, int log_nttSize, Modulus modulus, int batchSize, cudaStream_t stream)
{
    switch (log_nttSize)
    {
        case 12:
            NTTCore<<<dim3(8, 1, batchSize), dim3(64, 4), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 3, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 8, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 3, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 13:
            NTTCore<<<dim3(16, 1, batchSize), dim3(32, 8), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 4, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 16, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 4, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 14:
            NTTCore<<<dim3(32, 1, batchSize), dim3(16, 16), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 5, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 32, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 5, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 15:
            NTTCore<<<dim3(64, 1, batchSize), dim3(8, 32), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 6, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 64, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 6, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 16:
            NTTCore<<<dim3(128, 1, batchSize), dim3(4, 64), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 7, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 128, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 7, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 17:
            NTTCore<<<dim3(256, 1, batchSize), dim3(32, 8), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 4, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(16, 16, batchSize), dim3(32, 8), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 4, 4, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 256, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 8, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 18:
            NTTCore<<<dim3(512, 1, batchSize), dim3(32, 8), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 4, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(32, 16, batchSize), dim3(32, 8), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 4, 5, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 512, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 9, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 19:
            NTTCore<<<dim3(1024, 1, batchSize), dim3(16, 16), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 5, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(32, 32, batchSize), dim3(16, 16), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 5, 5, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 1024, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 10, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 20:
            NTTCore<<<dim3(2048, 1, batchSize), dim3(16, 16), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 5, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(64, 32, batchSize), dim3(8, 32), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 5, 6, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 2048, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 11, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 21:
            NTTCore<<<dim3(4096, 1, batchSize), dim3(8, 32), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 6, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(64, 64, batchSize), dim3(8, 32), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 6, 6, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 4096, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 12, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 22:
            NTTCore<<<dim3(8192, 1, batchSize), dim3(8, 32), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 6, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(128, 64, batchSize), dim3(4, 64), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 6, 7, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 8192, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 13, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 23:
            NTTCore<<<dim3(16384, 1, batchSize), dim3(4, 64), 512 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 8, 0, 7, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(128, 128, batchSize), dim3(4, 64), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 7, 7, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 16384, batchSize), dim3(256, 1), 512 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 8, 14, 9, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 24:
            NTTCore<<<dim3(16384, 1, batchSize), dim3(8, 64), 1024 * sizeof(uint64_t), stream>>>(input, output, rootUnityTable, modulus, 9, 0, 7, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(128, 128, batchSize), dim3(8, 64), 1024 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 9, 7, 7, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            NTTCore<<<dim3(1, 16384, batchSize), dim3(512, 1), 1024 * sizeof(uint64_t), stream>>>(output, output, rootUnityTable, modulus, 9, 14, 10, log_nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;

        default:
            std::cout << "Invalid log_nttSize" << std::endl;
            break;
    }
}


__host__ void GPU_INTT(uint64_t* input, uint64_t* output, uint64_t* rootUnityTable, int log_nttSize, uint64_t nttInv, Modulus modulus, int batchSize)
{
    switch (log_nttSize)
    {
        case 4:
            INTTCore<<<dim3(1, 1, batchSize), dim3(8, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 3, 3, 0, 4, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 5:
            INTTCore<<<dim3(1, 1, batchSize), dim3(16, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 4, 4, 0, 5, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 6:
            INTTCore<<<dim3(1, 1, batchSize), dim3(32, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 5, 5, 0, 6, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 7:
            INTTCore<<<dim3(1, 1, batchSize), dim3(64, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 6, 6, 0, 7, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 8:
            INTTCore<<<dim3(1, 1, batchSize), dim3(128, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 7, 7, 0, 8, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 9:
            INTTCore<<<dim3(1, 1, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 8, 8, 0, 9, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 10:
            INTTCore<<<dim3(1, 2, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 8, 9, 1, 9, log_nttSize, nttInv, false);
            CUDA_CHECK(cudaGetLastError());
            INTTCore<<<dim3(2, 1, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(output, output, rootUnityTable, modulus, 8, 0, 0, 1, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 11:
            INTTCore<<<dim3(1, 4, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 8, 10, 2, 9, log_nttSize, nttInv, false);
            CUDA_CHECK(cudaGetLastError());
            INTTCore<<<dim3(4, 1, batchSize), dim3(128, 2), 512 * sizeof(uint64_t)>>>(output, output, rootUnityTable, modulus, 8, 1, 0, 2, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 12:
            INTTCore<<<dim3(1, 8, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 8, 11, 3, 9, log_nttSize, nttInv, false);
            CUDA_CHECK(cudaGetLastError());
            INTTCore<<<dim3(8, 1, batchSize), dim3(64, 4), 512 * sizeof(uint64_t)>>>(output, output, rootUnityTable, modulus, 8, 2, 0, 3, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 13:
            INTTCore<<<dim3(1, 16, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 8, 12, 4, 9, log_nttSize, nttInv, false);
            CUDA_CHECK(cudaGetLastError());
            INTTCore<<<dim3(16, 1, batchSize), dim3(32, 8), 512 * sizeof(uint64_t)>>>(output, output, rootUnityTable, modulus, 8, 3, 0, 4, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 14:
            INTTCore<<<dim3(1, 32, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 8, 13, 5, 9, log_nttSize, nttInv, false);
            CUDA_CHECK(cudaGetLastError());
            INTTCore<<<dim3(32, 1, batchSize), dim3(16, 16), 512 * sizeof(uint64_t)>>>(output, output, rootUnityTable, modulus, 8, 4, 0, 5, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 15:
            INTTCore<<<dim3(1, 64, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 8, 14, 6, 9, log_nttSize, nttInv, false);
            CUDA_CHECK(cudaGetLastError());
            INTTCore<<<dim3(64, 1, batchSize), dim3(8, 32), 512 * sizeof(uint64_t)>>>(output, output, rootUnityTable, modulus, 8, 5, 0, 6, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 16:
            INTTCore<<<dim3(1, 128, batchSize), dim3(256, 1), 512 * sizeof(uint64_t)>>>(input, output, rootUnityTable, modulus, 8, 15, 7, 9, log_nttSize, nttInv, false);
            CUDA_CHECK(cudaGetLastError());
            INTTCore<<<dim3(128, 1, batchSize), dim3(4, 64), 512 * sizeof(uint64_t)>>>(output, output, rootUnityTable, modulus, 8, 6, 0, 7, log_nttSize, nttInv, true);
            CUDA_CHECK(cudaGetLastError());
            break;

        default:
            std::cout << "Invalid log_nttSize" << std::endl;
            break;
    }
}