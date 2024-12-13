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

#define WARP_SIZE 32
#define WMMA_M    16
#define WMMA_N    16
#define WMMA_K    16
#define WORD_IN   8
#define WORD_OUT  16
#define FRAG_NUM  8
#define COL_SIZE  16
#define FLAG      0


__global__ void tensorCore_2(int flag, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus, int nttSize)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int blockId = blockIdx.x;
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int warpNum = blockDim.y;
    int blockNum = gridDim.x;
    int subNum = gridDim.y;

    int colId = laneId / COL_SIZE;
    int rowId = laneId % COL_SIZE;
    int blockStep = blockNum * warpNum;
    int subStep = blockStep * COL_SIZE;
    int subSize = subStep * COL_SIZE;

    int index1 = blockId * warpNum + warpId + colId * FRAG_NUM * blockStep;

    int sharedIndex = rowId * WMMA_N + colId * FRAG_NUM;

    int inputIndex = batchId * nttSize + subId * subSize + rowId * subStep + index1;

    // 申请共享内存
    extern __shared__ uint64_t shared_mem[];
    auto shared_B = reinterpret_cast<uint8_t(*)[WORD_IN][WMMA_M * WMMA_K]>(shared_mem);
    auto shared_C = reinterpret_cast<int32_t(*)[2][WMMA_M * WMMA_K]>(shared_mem + (warpNum << 8));

    // 申请矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

    // 数据读取
    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t inputVal = input[inputIndex + i * blockStep];
        for (int j = 0; j < WORD_IN; j++)
        {
            shared_B[warpId][j][sharedIndex + i] = (inputVal >> (j * 8)) & 0xFF;
        }
    }

    // 数据填充
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + i * WMMA_M * WMMA_K, WMMA_K);
        wmma::load_matrix_sync(b_frag[i], shared_B[warpId][i], WMMA_N);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }

#pragma unroll
    for (int frag_i = 0; frag_i < FRAG_NUM; ++frag_i)
    {
        // 将计算结果拼接。
        uint64_t concate55[WORD_OUT / 4];
        concate55[0] = c_frag[0].x[frag_i];
        concate55[0] += static_cast<uint64_t>(c_frag[1].x[frag_i]) << 8;
        concate55[0] += static_cast<uint64_t>(c_frag[2].x[frag_i]) << 16;
        concate55[0] += static_cast<uint64_t>(c_frag[3].x[frag_i]) << 24;

        concate55[1] = c_frag[4 + 0].x[frag_i];
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 1].x[frag_i]) << 8;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 2].x[frag_i]) << 16;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 3].x[frag_i]) << 24;

        concate55[2] = c_frag[8 + 0].x[frag_i];
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 1].x[frag_i]) << 8;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 2].x[frag_i]) << 16;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 3].x[frag_i]) << 24;

        concate55[3] = c_frag[12 + 0].x[frag_i];
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 1].x[frag_i]) << 8;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 2].x[frag_i]) << 16;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 3].x[frag_i]) << 24;
        // 整理到32位
        uint32_t concate32[WORD_OUT / 4 + 1];
        concate32[0] = concate55[0] & 0xFFFFFFFF;
        for (int j = 1; j < WORD_OUT / 4; ++j)
        {
            concate55[j] += (concate55[j - 1] >> 32);
            concate32[j] = concate55[j] & 0xFFFFFFFF;
        }
        concate32[WORD_OUT / 4] = concate55[WORD_OUT / 4 - 1] >> 32;
        // 拼接成128位并取模
        __uint128_t a = (__uint128_t)concate32[1] | ((__uint128_t)concate32[2] << 32) | ((__uint128_t)concate32[3] << 64) | ((__uint128_t)concate32[4] << 96);
        a = Barrett64_gpu::reduction(a, modulus);
        a = (a << 32) + concate32[0];
        uint64_t temp = Barrett64_gpu::reduction(a, modulus);
        // 写回
        c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
        c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
    }
    wmma::store_matrix_sync(shared_C[warpId][0], c_frag[0], WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(shared_C[warpId][1], c_frag[1], WMMA_N, wmma::mem_row_major);

    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t temp = shared_C[warpId][0][sharedIndex + i];
        temp += ((uint64_t)shared_C[warpId][1][sharedIndex + i]) << 32;
        temp = Barrett64_gpu::mult(temp, factorTable[rowId * (index1 + i * blockStep) * subNum], modulus);
        int tempIndex = rowId + (colId * FRAG_NUM + i) * WMMA_N;
        for (int j = 0; j < WORD_IN; j++)
        {
            shared_B[warpId][j][tempIndex] = (temp >> (j * 8)) & 0xFF;
        }
    }
    // 数据填充
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(b_frag[i], shared_B[warpId][i], WMMA_N);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }
#pragma unroll
    for (int frag_i = 0; frag_i < FRAG_NUM; ++frag_i)
    {
        // 将计算结果拼接。
        uint64_t concate55[WORD_OUT / 4];
        concate55[0] = c_frag[0].x[frag_i];
        concate55[0] += static_cast<uint64_t>(c_frag[1].x[frag_i]) << 8;
        concate55[0] += static_cast<uint64_t>(c_frag[2].x[frag_i]) << 16;
        concate55[0] += static_cast<uint64_t>(c_frag[3].x[frag_i]) << 24;

        concate55[1] = c_frag[4 + 0].x[frag_i];
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 1].x[frag_i]) << 8;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 2].x[frag_i]) << 16;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 3].x[frag_i]) << 24;

        concate55[2] = c_frag[8 + 0].x[frag_i];
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 1].x[frag_i]) << 8;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 2].x[frag_i]) << 16;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 3].x[frag_i]) << 24;

        concate55[3] = c_frag[12 + 0].x[frag_i];
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 1].x[frag_i]) << 8;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 2].x[frag_i]) << 16;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 3].x[frag_i]) << 24;
        // 整理到32位
        uint32_t concate32[WORD_OUT / 4 + 1];
        concate32[0] = concate55[0] & 0xFFFFFFFF;
        for (int j = 1; j < WORD_OUT / 4; ++j)
        {
            concate55[j] += (concate55[j - 1] >> 32);
            concate32[j] = concate55[j] & 0xFFFFFFFF;
        }
        concate32[WORD_OUT / 4] = concate55[WORD_OUT / 4 - 1] >> 32;
        // 拼接成128位并取模
        __uint128_t a = (__uint128_t)concate32[1] | ((__uint128_t)concate32[2] << 32) | ((__uint128_t)concate32[3] << 64) | ((__uint128_t)concate32[4] << 96);
        a = Barrett64_gpu::reduction(a, modulus);
        a = (a << 32) + concate32[0];
        uint64_t temp = Barrett64_gpu::reduction(a, modulus);
        // 写回
        c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
        c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
    }
    wmma::store_matrix_sync(shared_C[warpId][0], c_frag[0], WMMA_N, wmma::mem_col_major);
    wmma::store_matrix_sync(shared_C[warpId][1], c_frag[1], WMMA_N, wmma::mem_col_major);
    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t temp = shared_C[warpId][0][sharedIndex + i];
        temp += ((uint64_t)shared_C[warpId][1][sharedIndex + i]) << 32;
        temp = Barrett64_gpu::mult(temp, factorTable[(colId * FRAG_NUM + i) * (blockId * warpNum + warpId) * 16 * subNum], modulus);
        output[inputIndex + i * blockStep] = temp;
    }
}

__global__ void tensorCore_2(const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus, int nttSize)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;

    int blockId = blockIdx.x;
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int warpNum = blockDim.y;
    int blockNum = gridDim.x;
    int subNum = gridDim.y;

    int colId = laneId / COL_SIZE;
    int rowId = laneId % COL_SIZE;

    int blockStep = blockNum * warpNum * WMMA_N;
    int subSize = blockStep * COL_SIZE;

    int sharedIndex = rowId * WMMA_N + colId * FRAG_NUM;
    int inputIndex = batchId * nttSize + subId * subSize + rowId * blockStep + (blockId * warpNum + warpId) * WMMA_N + colId * FRAG_NUM;
   
    int scale = subNum * rowId;
    int factorIndex = ((blockId * warpNum + warpId) * WMMA_N + colId * FRAG_NUM) * scale;

    extern __shared__ uint64_t shared_mem[];
    auto shared_B = reinterpret_cast<uint8_t(*)[WORD_IN][WMMA_M * WMMA_K]>(shared_mem);
    auto shared_C = reinterpret_cast<int32_t(*)[2][WMMA_M * WMMA_K]>(shared_mem + (warpNum << 8));

    // 申请矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t inputVal = input[inputIndex + i];
        for (int j = 0; j < WORD_IN; j++)
        {
            shared_B[warpId][j][sharedIndex + i] = (inputVal >> (j * 8)) & 0xFF;
        }
    }

    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + i * WMMA_M * WMMA_K, WMMA_K);
        wmma::load_matrix_sync(b_frag[i], shared_B[warpId][i], WMMA_N);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }


    for (int frag_i = 0; frag_i < FRAG_NUM; ++frag_i)
    {
        // 将计算结果拼接。
        uint64_t concate55[WORD_OUT / 4];
        concate55[0] = c_frag[0].x[frag_i];
        concate55[0] += static_cast<uint64_t>(c_frag[1].x[frag_i]) << 8;
        concate55[0] += static_cast<uint64_t>(c_frag[2].x[frag_i]) << 16;
        concate55[0] += static_cast<uint64_t>(c_frag[3].x[frag_i]) << 24;

        concate55[1] = c_frag[4 + 0].x[frag_i];
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 1].x[frag_i]) << 8;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 2].x[frag_i]) << 16;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 3].x[frag_i]) << 24;

        concate55[2] = c_frag[8 + 0].x[frag_i];
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 1].x[frag_i]) << 8;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 2].x[frag_i]) << 16;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 3].x[frag_i]) << 24;

        concate55[3] = c_frag[12 + 0].x[frag_i];
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 1].x[frag_i]) << 8;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 2].x[frag_i]) << 16;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 3].x[frag_i]) << 24;
        // 整理到32位
        uint32_t concate32[WORD_OUT / 4 + 1];
        concate32[0] = concate55[0] & 0xFFFFFFFF;
        for (int j = 1; j < WORD_OUT / 4; ++j)
        {
            concate55[j] += (concate55[j - 1] >> 32);
            concate32[j] = concate55[j] & 0xFFFFFFFF;
        }
        concate32[WORD_OUT / 4] = concate55[WORD_OUT / 4 - 1] >> 32;
        // 拼接成128位并取模
        __uint128_t a = (__uint128_t)concate32[1] | ((__uint128_t)concate32[2] << 32) | ((__uint128_t)concate32[3] << 64) | ((__uint128_t)concate32[4] << 96);
        a = Barrett64_gpu::reduction(a, modulus);
        a = (a << 32) + concate32[0];
        uint64_t temp = Barrett64_gpu::reduction(a, modulus);
        ;
        // 写回
        c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
        c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
    }
    wmma::store_matrix_sync(shared_C[warpId][0], c_frag[0], WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(shared_C[warpId][1], c_frag[1], WMMA_N, wmma::mem_row_major);

    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t temp = shared_C[warpId][0][sharedIndex + i];
        temp += ((uint64_t)shared_C[warpId][1][sharedIndex + i]) << 32;
        temp = Barrett64_gpu::mult(temp, factorTable[factorIndex + i * scale], modulus);
        output[inputIndex + i] = temp;
    }
}

__global__ void tensorCore_2(int minorSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, const uint8_t* minorMatrix,
                                Modulus modulus, int nttSize)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int blockId = blockIdx.x;
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int warpNum = blockDim.y;
    int blockNum = gridDim.x;

    int tileNum = COL_SIZE / minorSize;
    int tileSize = minorSize * COL_SIZE;
    int warpSize = tileNum * tileSize;
    int blockSize = warpSize * warpNum;
    int subSize = blockSize * blockNum;

    int colId = laneId / COL_SIZE;
    int rowId = laneId % COL_SIZE;

    int sharedIndex = rowId * WMMA_N + colId * FRAG_NUM;
    int inputIndex = batchId * nttSize + subId * subSize + blockId * blockSize + warpId * warpSize + rowId * minorSize + colId * (tileNum / 2) * tileSize +
                     colId * FRAG_NUM * (minorSize / COL_SIZE);

    int scale = nttSize / tileSize;

    extern __shared__ uint64_t shared_mem[];
    auto shared_B = reinterpret_cast<uint8_t(*)[WORD_IN][WMMA_M * WMMA_K]>(shared_mem);
    auto shared_C = reinterpret_cast<int32_t(*)[2][WMMA_M * WMMA_K]>(shared_mem + (warpNum << 8));

    // 申请矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

#pragma unroll
    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t inputVal = input[inputIndex + i % minorSize + i / minorSize * tileSize];
        for (int j = 0; j < WORD_IN; j++)
        {
            shared_B[warpId][j][sharedIndex + i] = (inputVal >> (j * 8)) & 0xFF;
        }
    }

    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    for (int i = 0; i < WORD_IN; i++)
    {

        wmma::load_matrix_sync(a_frag[i], majorMatrix + i * WMMA_M * WMMA_K, WMMA_K);
        wmma::load_matrix_sync(b_frag[i], shared_B[warpId][i], WMMA_N);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }

#pragma unroll
    for (int frag_i = 0; frag_i < FRAG_NUM; ++frag_i)
    {
        // 将计算结果拼接。
        uint64_t concate55[WORD_OUT / 4];
        concate55[0] = c_frag[0].x[frag_i];
        concate55[0] += static_cast<uint64_t>(c_frag[1].x[frag_i]) << 8;
        concate55[0] += static_cast<uint64_t>(c_frag[2].x[frag_i]) << 16;
        concate55[0] += static_cast<uint64_t>(c_frag[3].x[frag_i]) << 24;

        concate55[1] = c_frag[4 + 0].x[frag_i];
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 1].x[frag_i]) << 8;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 2].x[frag_i]) << 16;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 3].x[frag_i]) << 24;

        concate55[2] = c_frag[8 + 0].x[frag_i];
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 1].x[frag_i]) << 8;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 2].x[frag_i]) << 16;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 3].x[frag_i]) << 24;

        concate55[3] = c_frag[12 + 0].x[frag_i];
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 1].x[frag_i]) << 8;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 2].x[frag_i]) << 16;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 3].x[frag_i]) << 24;
        // 整理到32位
        uint32_t concate32[WORD_OUT / 4 + 1];
        concate32[0] = concate55[0] & 0xFFFFFFFF;
        for (int j = 1; j < WORD_OUT / 4; ++j)
        {
            concate55[j] += (concate55[j - 1] >> 32);
            concate32[j] = concate55[j] & 0xFFFFFFFF;
        }
        concate32[WORD_OUT / 4] = concate55[WORD_OUT / 4 - 1] >> 32;
        // 拼接成128位并取模
        __uint128_t a = (__uint128_t)concate32[1] | ((__uint128_t)concate32[2] << 32) | ((__uint128_t)concate32[3] << 64) | ((__uint128_t)concate32[4] << 96);
        a = Barrett64_gpu::reduction(a, modulus);
        a = (a << 32) + concate32[0];
        uint64_t temp = Barrett64_gpu::reduction(a, modulus);
        // 写回
        c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
        c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
    }
    wmma::store_matrix_sync(shared_C[warpId][0], c_frag[0], WMMA_N, wmma::mem_col_major);
    wmma::store_matrix_sync(shared_C[warpId][1], c_frag[1], WMMA_N, wmma::mem_col_major);

    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t temp = shared_C[warpId][0][sharedIndex + i];
        temp += ((uint64_t)shared_C[warpId][1][sharedIndex + i]) << 32;
        temp = Barrett64_gpu::mult(temp, factorTable[rowId * ((colId * FRAG_NUM + i) % minorSize) * scale], modulus);
        for (int j = 0; j < WORD_IN; j++)
        {
            shared_B[warpId][j][sharedIndex + i] = (temp >> (j * 8)) & 0xFF;
        }
    }


    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], minorMatrix + i * WMMA_M * WMMA_K, WMMA_K);
        wmma::load_matrix_sync(b_frag[i], shared_B[warpId][i], WMMA_N);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }

    // 约简

    for (int frag_i = 0; frag_i < FRAG_NUM; ++frag_i)
    {
        // 将计算结果拼接。
        uint64_t concate55[WORD_OUT / 4];
        concate55[0] = c_frag[0].x[frag_i];
        concate55[0] += static_cast<uint64_t>(c_frag[1].x[frag_i]) << 8;
        concate55[0] += static_cast<uint64_t>(c_frag[2].x[frag_i]) << 16;
        concate55[0] += static_cast<uint64_t>(c_frag[3].x[frag_i]) << 24;

        concate55[1] = c_frag[4 + 0].x[frag_i];
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 1].x[frag_i]) << 8;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 2].x[frag_i]) << 16;
        concate55[1] += static_cast<uint64_t>(c_frag[4 + 3].x[frag_i]) << 24;

        concate55[2] = c_frag[8 + 0].x[frag_i];
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 1].x[frag_i]) << 8;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 2].x[frag_i]) << 16;
        concate55[2] += static_cast<uint64_t>(c_frag[8 + 3].x[frag_i]) << 24;

        concate55[3] = c_frag[12 + 0].x[frag_i];
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 1].x[frag_i]) << 8;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 2].x[frag_i]) << 16;
        concate55[3] += static_cast<uint64_t>(c_frag[12 + 3].x[frag_i]) << 24;
        // 整理到32位
        uint32_t concate32[WORD_OUT / 4 + 1];
        concate32[0] = concate55[0] & 0xFFFFFFFF;
        for (int j = 1; j < WORD_OUT / 4; ++j)
        {
            concate55[j] += (concate55[j - 1] >> 32);
            concate32[j] = concate55[j] & 0xFFFFFFFF;
        }
        concate32[WORD_OUT / 4] = concate55[WORD_OUT / 4 - 1] >> 32;
        // 拼接成128位并取模
        __uint128_t a = (__uint128_t)concate32[1] | ((__uint128_t)concate32[2] << 32) | ((__uint128_t)concate32[3] << 64) | ((__uint128_t)concate32[4] << 96);
        a = Barrett64_gpu::reduction(a, modulus);
        a = (a << 32) + concate32[0];
        uint64_t temp = Barrett64_gpu::reduction(a, modulus);
        // 写回
        c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
        c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
    }
    wmma::store_matrix_sync(shared_C[warpId][0], c_frag[0], WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(shared_C[warpId][1], c_frag[1], WMMA_N, wmma::mem_row_major);

    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t temp = shared_C[warpId][0][sharedIndex + i];
        temp += ((uint64_t)shared_C[warpId][1][sharedIndex + i]) << 32;
        temp=Barrett64_gpu::mult(temp,factorTable[rowId*((colId*FRAG_NUM+i)%minorSize)*scale],modulus);
        output[inputIndex + i % minorSize + i / minorSize * tileSize] = temp;
    }
}



__host__ void GPU_TensorNTT_2(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize, cudaStream_t stream)
{
    switch (log_nttSize)
    {
        case 12:
            tensorCore_2<<<dim3(4, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(input, output, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(4, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(16, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 13:
            tensorCore_2<<<dim3(8, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(8, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 14:
            tensorCore_2<<<dim3(16, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(16, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 15:
            tensorCore_2<<<dim3(32, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(32, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(8, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 16:
            tensorCore_2<<<dim3(64, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(64, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(16, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 17:
            tensorCore_2<<<dim3(128, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(1, 256, batchSize), dim3(32, 2), sizeof(uint64_t) << 10, stream>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(128, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 18:
            tensorCore_2<<<dim3(256, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(1, 256, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(256, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 19:
            tensorCore_2<<<dim3(512, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(2, 256, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(512, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(8, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 20:
            tensorCore_2<<<dim3(1024, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(4, 256, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(1024, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(16, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 21:
            tensorCore_2<<<dim3(2048, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(8, 256, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(2048, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 22:
            tensorCore_2<<<dim3(4096, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(16, 256, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(4096, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 23:
            tensorCore_2<<<dim3(8192, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(32, 256, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(8192, 1, batchSize), dim3(32, 4), sizeof(uint64_t) << 11, stream>>>(8, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 24:

            tensorCore_2<<<dim3(8192, 1, batchSize), dim3(32, 8), sizeof(uint64_t) << 12, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_2<<<dim3(32, 256, batchSize), dim3(32, 8), sizeof(uint64_t) << 12, stream>>>(FLAG, input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());

            tensorCore_2<<<dim3(8192, 1, batchSize), dim3(32, 8), sizeof(uint64_t) << 12, stream>>>(16, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);

            CUDA_CHECK(cudaGetLastError());
            break;
    }
}
