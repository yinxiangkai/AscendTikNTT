#include "common.cuh"
#include "modular.cuh"
#include "tensor_ntt.cuh"
#include <climits>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>
#include <sys/types.h>
using namespace nvcuda;

#define WARP_SIZE     32
#define WMMA_M        32
#define WMMA_N        8
#define WMMA_K        16
#define WORD_IN       8
#define WORD_OUT      8
#define FRAG_NUM      8
#define COL_SIZE      32
#define FLAG          0
#define MATRIX_OFFSET 512

__global__ void __launch_bounds__(1024) tensorCore_8(const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                                                        Modulus modulus, int nttSize, int logStep)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int blockId = blockIdx.x;
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int log_rowSize = logStep+5;
    int log_subSize = log_rowSize+5;

    int scale = gridDim.y;

    int lineId = laneId & 3;
    int rowId = (laneId >> 2) + (lineId << 3);
    int colId;

    if (logStep == 0)
    {
        colId = (blockId << 5) + warpId;
    }
    else
    {
        colId = blockId + (warpId << logStep);
    }

    // 计算输入输出偏移
    int offset = batchId * nttSize + (subId <<log_subSize);
    const uint8_t* inputindex = reinterpret_cast<const uint8_t*>(input + offset + colId);

    // 申请共享内存
    __shared__ uint64_t shared_mem[32 * 32];
    __shared__ uint64_t temp[32][32][4];

    // 申请wmma片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

    // 数据填充
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + i * MATRIX_OFFSET, WMMA_K);
    }

    wmma::load_matrix_sync(b_frag, inputindex, (WMMA_N<<log_rowSize));

    // 计算
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::mma_sync(c_frag[i], a_frag[i], b_frag, c_frag[i]);
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + (i + 8) * MATRIX_OFFSET, WMMA_K);
    }

    wmma::load_matrix_sync(b_frag, inputindex + (WMMA_N << log_rowSize), (WMMA_N << log_rowSize));

    // 计算
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::mma_sync(c_frag[i], a_frag[i], b_frag, c_frag[i]);
    }

    // 约简

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int index1 = i << 1;
        int index2 = (i << 1) + 1;
        uint64_t concate64[2];

        concate64[0] = (uint64_t)c_frag[0].x[index1];
        concate64[0] += (uint64_t)(c_frag[0].x[index2] + c_frag[1].x[index1]) << 8;
        concate64[0] += (uint64_t)(c_frag[1].x[index2] + c_frag[2].x[index1]) << 16;
        concate64[0] += (uint64_t)(c_frag[2].x[index2] + c_frag[3].x[index1]) << 24;
        concate64[0] += (uint64_t)(c_frag[3].x[index2]) << 32;

        concate64[1] = (uint64_t)(c_frag[4].x[index1]);
        concate64[1] += (uint64_t)(c_frag[4].x[index2] + c_frag[5].x[index1]) << 8;
        concate64[1] += (uint64_t)(c_frag[5].x[index2] + c_frag[6].x[index1]) << 16;
        concate64[1] += (uint64_t)(c_frag[6].x[index2] + c_frag[7].x[index1]) << 24;
        concate64[1] += (uint64_t)(c_frag[7].x[index2]) << 32;

        __uint128_t a = ((__uint128_t)concate64[0]) + ((__uint128_t)concate64[1] << 32);
        temp[warpId][(laneId >> 2) + (i << 3)][lineId] = Barrett64_gpu::reduction(a, modulus);
    }
    __uint128_t a = (__uint128_t)temp[warpId][rowId][0] + ((__uint128_t)temp[warpId][rowId][1] << 16) + ((__uint128_t)temp[warpId][rowId][2] << 32) +
                    ((__uint128_t)temp[warpId][rowId][3] << 48);

    uint64_t result = Barrett64_gpu::reduction(a, modulus);
    result = Barrett64_gpu::mult(result, factorTable[scale * rowId * colId], modulus);



    if (logStep== 0)
    {
        output[offset + (rowId <<log_rowSize) + colId] = result;
    }
    else
    {
        shared_mem[warpId * COL_SIZE + rowId] = result;
        __syncthreads();

        // 数据填充
        for (int i = 0; i < WORD_OUT; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }

        // 读取输入
        wmma::load_matrix_sync(b_frag, reinterpret_cast<const uint8_t*>(shared_mem + warpId + WMMA_K * 32), 256);

        // 计算
        for (int i = 0; i < WORD_OUT; i++)
        {
            wmma::mma_sync(c_frag[i], a_frag[i], b_frag, c_frag[i]);
        }

        // 读取输入
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + i * MATRIX_OFFSET, WMMA_K);
        }
        wmma::load_matrix_sync(b_frag, reinterpret_cast<const uint8_t*>(shared_mem + warpId), 256);
        // 计算
        for (int i = 0; i < WORD_OUT; i++)
        {
            wmma::mma_sync(c_frag[i], a_frag[i], b_frag, c_frag[i]);
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int index1 = i << 1;
            int index2 = (i << 1) + 1;
            uint64_t concate64[2];

            concate64[0] = (uint64_t)c_frag[0].x[index1];
            concate64[0] += (uint64_t)(c_frag[0].x[index2] + c_frag[1].x[index1]) << 8;
            concate64[0] += (uint64_t)(c_frag[1].x[index2] + c_frag[2].x[index1]) << 16;
            concate64[0] += (uint64_t)(c_frag[2].x[index2] + c_frag[3].x[index1]) << 24;
            concate64[0] += (uint64_t)(c_frag[3].x[index2]) << 32;

            concate64[1] = (uint64_t)(c_frag[4].x[index1]);
            concate64[1] += (uint64_t)(c_frag[4].x[index2] + c_frag[5].x[index1]) << 8;
            concate64[1] += (uint64_t)(c_frag[5].x[index2] + c_frag[6].x[index1]) << 16;
            concate64[1] += (uint64_t)(c_frag[6].x[index2] + c_frag[7].x[index1]) << 24;
            concate64[1] += (uint64_t)(c_frag[7].x[index2]) << 32;

            __uint128_t a = ((__uint128_t)concate64[0]) + ((__uint128_t)concate64[1] << 32);
            temp[warpId][(laneId >> 2) + (i << 3)][lineId] = Barrett64_gpu::reduction(a, modulus);
        }
        a = (__uint128_t)temp[warpId][rowId][0] + ((__uint128_t)temp[warpId][rowId][1] << 16) + ((__uint128_t)temp[warpId][rowId][2] << 32) +
            ((__uint128_t)temp[warpId][rowId][3] << 48);

        result = Barrett64_gpu::reduction(a, modulus);
        result = Barrett64_gpu::mult(result, factorTable[COL_SIZE * scale * rowId * blockId], modulus);

        output[offset + (warpId <<log_rowSize) + blockId + (rowId << logStep)] = result;
    }
}



__global__ void __launch_bounds__(1024) tensorCore_8(int log_minorSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix,
                                                        uint64_t* factorTable, const uint8_t* minorMatrix, Modulus modulus, int nttSize)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int blockId = blockIdx.x;
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int minorSize = 1 << log_minorSize;
    int tileId = warpId >> log_minorSize;
    int colId = warpId & (minorSize - 1);

    // int warpNum = blockDim.y;
    int blockSize = 1024;
    int tileSize = minorSize << 5;
    int tileNum = COL_SIZE >> log_minorSize;

    // 64bit
    int minorStep = minorSize << 3;
    int scale = gridDim.y * gridDim.x * tileNum;

    int offset = batchId * nttSize + (subId * gridDim.x + blockId) * blockSize;
    const uint8_t* inputindex = reinterpret_cast<const uint8_t*>(input + offset + tileId * tileSize + colId);

    // 约简用
    int lineId = laneId & 3;
    int rowId = (laneId >> 2) + (lineId << 3);

    __shared__ uint64_t shared_mem[32 * 32];

    __shared__ uint64_t temp[32][32][4];

    // 申请wmma片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

    // 数据填充
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + i * MATRIX_OFFSET, WMMA_K);
    }
    wmma::load_matrix_sync(b_frag, inputindex, minorStep);

    // 计算
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::mma_sync(c_frag[i], a_frag[i], b_frag, c_frag[i]);
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + (i + 8) * MATRIX_OFFSET, WMMA_K);
    }
    wmma::load_matrix_sync(b_frag, inputindex + WMMA_K * minorStep, minorStep);

    // 计算
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::mma_sync(c_frag[i], a_frag[i], b_frag, c_frag[i]);
    }

    // 约简

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int index1 = i << 1;
        int index2 = (i << 1) + 1;
        uint64_t concate64[2];

        concate64[0] = (uint64_t)c_frag[0].x[index1];
        concate64[0] += (uint64_t)(c_frag[0].x[index2] + c_frag[1].x[index1]) << 8;
        concate64[0] += (uint64_t)(c_frag[1].x[index2] + c_frag[2].x[index1]) << 16;
        concate64[0] += (uint64_t)(c_frag[2].x[index2] + c_frag[3].x[index1]) << 24;
        concate64[0] += (uint64_t)(c_frag[3].x[index2]) << 32;

        concate64[1] = (uint64_t)(c_frag[4].x[index1]);
        concate64[1] += (uint64_t)(c_frag[4].x[index2] + c_frag[5].x[index1]) << 8;
        concate64[1] += (uint64_t)(c_frag[5].x[index2] + c_frag[6].x[index1]) << 16;
        concate64[1] += (uint64_t)(c_frag[6].x[index2] + c_frag[7].x[index1]) << 24;
        concate64[1] += (uint64_t)(c_frag[7].x[index2]) << 32;

        __uint128_t a = ((__uint128_t)concate64[0]) + ((__uint128_t)concate64[1] << 32);
        temp[warpId][(laneId >> 2) + (i << 3)][lineId] = Barrett64_gpu::reduction(a, modulus);
    }
    __uint128_t a = (__uint128_t)temp[warpId][rowId][0] + ((__uint128_t)temp[warpId][rowId][1] << 16) + ((__uint128_t)temp[warpId][rowId][2] << 32) +
                    ((__uint128_t)temp[warpId][rowId][3] << 48);

    uint64_t result = Barrett64_gpu::reduction(a, modulus);

    result = Barrett64_gpu::mult(result, factorTable[scale * rowId * colId], modulus);
    shared_mem[warpId * COL_SIZE + rowId] = result;
    __syncthreads();

    /*
     * 行NTT
     */

    // 数据填充
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], minorMatrix + i * MATRIX_OFFSET, WMMA_K);
    }
    wmma::load_matrix_sync(b_frag, reinterpret_cast<const uint8_t*>(shared_mem + warpId), 256);

    // 计算
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::mma_sync(c_frag[i], a_frag[i], b_frag, c_frag[i]);
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], minorMatrix + (i + 8) * MATRIX_OFFSET, WMMA_K);
    }

    wmma::load_matrix_sync(b_frag, reinterpret_cast<const uint8_t*>(shared_mem + 512 + warpId), 256);

    // 计算
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::mma_sync(c_frag[i], a_frag[i], b_frag, c_frag[i]);
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int index1 = i << 1;
        int index2 = (i << 1) + 1;
        uint64_t concate64[2];

        concate64[0] = (uint64_t)c_frag[0].x[index1];
        concate64[0] += (uint64_t)(c_frag[0].x[index2] + c_frag[1].x[index1]) << 8;
        concate64[0] += (uint64_t)(c_frag[1].x[index2] + c_frag[2].x[index1]) << 16;
        concate64[0] += (uint64_t)(c_frag[2].x[index2] + c_frag[3].x[index1]) << 24;
        concate64[0] += (uint64_t)(c_frag[3].x[index2]) << 32;

        concate64[1] = (uint64_t)(c_frag[4].x[index1]);
        concate64[1] += (uint64_t)(c_frag[4].x[index2] + c_frag[5].x[index1]) << 8;
        concate64[1] += (uint64_t)(c_frag[5].x[index2] + c_frag[6].x[index1]) << 16;
        concate64[1] += (uint64_t)(c_frag[6].x[index2] + c_frag[7].x[index1]) << 24;
        concate64[1] += (uint64_t)(c_frag[7].x[index2]) << 32;

        __uint128_t a = ((__uint128_t)concate64[0]) + ((__uint128_t)concate64[1] << 32);
        temp[warpId][(laneId >> 2) + (i << 3)][lineId] = Barrett64_gpu::reduction(a, modulus);
    }
    a = (__uint128_t)temp[warpId][rowId][0] + ((__uint128_t)temp[warpId][rowId][1] << 16) + ((__uint128_t)temp[warpId][rowId][2] << 32) +
        ((__uint128_t)temp[warpId][rowId][3] << 48);

    result = Barrett64_gpu::reduction(a, modulus);
    output[offset + warpId * COL_SIZE + rowId] = result;
}


__host__ void GPU_TensorNTT_8(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable,
                              int log_nttSize, Modulus modulus, int batchSize, int nttSize)
{
    switch (log_nttSize)
    {

        case 12:
            tensorCore_8<<<dim3(4, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 0);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(4, 1, batchSize), dim3(32, 4)>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;

        case 13:
            tensorCore_8<<<dim3(8, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 0);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(8, 1, batchSize), dim3(32, 4)>>>(3, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());

            break;
        case 14:
            tensorCore_8<<<dim3(16, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 0);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(16, 1, batchSize), dim3(32, 4)>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());

            break;
        case 15:
            tensorCore_8<<<dim3(32, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 0);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(32, 1, batchSize), dim3(32, 4)>>>(5, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 16:
            tensorCore_8<<<dim3(64, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 6);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(64, 1, batchSize), dim3(32, 4)>>>(1, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 17:
            tensorCore_8<<<dim3(128, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 7);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(128, 1, batchSize), dim3(32, 4)>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 18:
            tensorCore_8<<<dim3(256, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 8);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(256, 1, batchSize), dim3(32, 32)>>>(3, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 19:
            tensorCore_8<<<dim3(512, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 9);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(512, 1, batchSize), dim3(32, 32)>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 20:
            tensorCore_8<<<dim3(1024, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 10);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(1024, 1, batchSize), dim3(32, 32)>>>(5, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 21:
            tensorCore_8<<<dim3(2048, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 11);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(64, 32, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 6);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(2048, 1, batchSize), dim3(32, 32)>>>(1, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 22:
            tensorCore_8<<<dim3(4096, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 12);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(128, 32, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 7);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(4096, 1, batchSize), dim3(32, 32)>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            break;
        case 23:
            tensorCore_8<<<dim3(8192, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 13);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(256, 32, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 8);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(8192, 1, batchSize), dim3(32, 32)>>>(3, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            break;
        case 24:
            tensorCore_8<<<dim3(16384, 1, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 14);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(512, 32, batchSize), dim3(32, 32)>>>(input, input, majorMatrix, factorTable, modulus, nttSize, 9);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_8<<<dim3(16384, 1, batchSize), dim3(32, 32)>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            break;
    }
}
