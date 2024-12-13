#include "common.cuh"
#include "modular.cuh"
#include "tensor_ntt.cuh"
#include <climits>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <mma.h>
#include <semaphore.h>
#include <sys/types.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define WMMA_M    32
#define WMMA_N    8
#define WMMA_K    16
#define WORD_IN   8
#define WORD_OUT  16
#define FRAG_NUM  8
#define COL_SIZE  32
#define MATRIX_OFFSET 512

__global__ void __launch_bounds__(1024)
    tensorCore_7A(int logStep, int log_nttSize, int log_subNum, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int blockId = blockIdx.x;
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int log_subSize = log_nttSize - log_subNum;
    int log_rowSize = log_subSize - 5;

    int scale = gridDim.y;

    int b_colId = (laneId >> 2);
    int b_rowId = ((laneId & 3) << 2);
    int c_colId = (laneId & 3)<<1;
    int c_rowId = (laneId >> 2);
    int out_colId;
    int out_rowId;

    int factor_colId;
    int factor_rowId;
    // int shared_colId;
    int shared_rowId;


    // 申请共享内存
    __shared__ uint64_t shared_C[32 * 32];
    // 申请wmma片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];
    if (logStep == 0)
    {
        int addrOff = (batchId << log_nttSize) + (subId << log_subSize) ;
        int inputIndex = addrOff + (blockId << 5) + (b_rowId << log_rowSize) + (warpId << 3) + b_colId;

        for (int i = 0; i < WORD_OUT; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            uint64_t inputVal = input[inputIndex + (i << log_rowSize)];

            for (int j = 0; j < WORD_IN; j++)
            {
                b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
            }
        }

        // 读取输入
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + i * MATRIX_OFFSET, WMMA_K);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            uint64_t inputVal = input[inputIndex + ((i+16) << log_rowSize)];
            for (int j = 0; j < WORD_IN; j++)
            {
                b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
            }
        }

        // 读取输入
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + (i+8) * MATRIX_OFFSET, WMMA_K);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }

        // 约简
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
            uint64_t result = Barrett64_gpu::reduction(a, modulus);
            factor_colId = (blockId << 5) + (warpId<<3)+c_colId+(frag_i&1);
            factor_rowId = c_rowId + ((frag_i>>1)<<3);
            result = Barrett64_gpu::mult(result, factorTable[factor_rowId * factor_colId * scale], modulus);
            output[addrOff + factor_colId + (factor_rowId << log_rowSize)] = result;
        }
    }
    else
    {
        int addrOff = (batchId << log_nttSize) + (subId << log_subSize);
        int inputIndex = addrOff + blockId + (b_rowId << log_rowSize) + (((warpId << 3) + b_colId) << logStep);

        for (int i = 0; i < WORD_OUT; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            uint64_t inputVal = input[inputIndex + (i << log_rowSize)];

            for (int j = 0; j < WORD_IN; j++)
            {
                b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
            }
        }

        // 读取输入
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + i * MATRIX_OFFSET, WMMA_K);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            uint64_t inputVal = input[inputIndex + ((i + 16) << log_rowSize)];
            for (int j = 0; j < WORD_IN; j++)
            {
                b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
            }
        }

        // 读取输入
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + (i + 8) * MATRIX_OFFSET, WMMA_K);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }
// 约简
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
            uint64_t result = Barrett64_gpu::reduction(a, modulus);
            shared_rowId = (warpId << 3) + c_colId +  (frag_i & 1);
            factor_colId = blockId + (shared_rowId <<logStep) ;
            factor_rowId = c_rowId + ((frag_i >> 1) << 3);
            result = Barrett64_gpu::mult(result, factorTable[factor_rowId * factor_colId * scale], modulus);
            shared_C[factor_rowId+(shared_rowId<<5)] = result;
        }

        for (int i = 0; i < WORD_OUT; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            uint64_t inputVal = shared_C[((b_rowId + i) << 5) + (warpId << 3) + b_colId];

            for (int j = 0; j < WORD_IN; j++)
            {
                b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
            }
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }

        // 读取输入
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + i * MATRIX_OFFSET, WMMA_K);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            uint64_t inputVal = shared_C[((b_rowId + i + 16) << 5) + (warpId << 3) + b_colId];


            for (int j = 0; j < WORD_IN; j++)
            {
                b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
            }
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }
        // 约简
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
            uint64_t result = Barrett64_gpu::reduction(a, modulus);
            factor_colId = blockId;
            factor_rowId = (warpId << 3) + c_colId + (frag_i & 1);
            out_rowId = factor_rowId;
            out_colId = blockId + ((c_rowId + ((frag_i >> 1) << 3))<<logStep);
            result = Barrett64_gpu::mult(result, factorTable[factor_rowId * factor_colId * scale], modulus);
            output[addrOff + out_colId + (out_rowId << log_rowSize)] = result;
        }
    }
}
__global__ void tensorCore_7B(int log_minorSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                                                     const uint8_t* minorMatrix, Modulus modulus, int nttSize)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int blockId = blockIdx.x;
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int unitId = subId*gridDim.x + blockId;
    int log_unitSize = 10;
    int minorSize = 1 << log_minorSize;
    int log_tileSize = log_minorSize + 5;
    int log_tileNum = 5 - log_minorSize;

    int b_colId = (laneId >> 2);
    int b_rowId = ((laneId & 3) << 2);
    int c_colId = (laneId & 3) << 1;
    int c_rowId = (laneId >> 2);

    int scale = gridDim.y * (gridDim.x <<log_tileNum);
    int tileId = ((warpId << 3) + b_colId) >> log_minorSize;

    int in_colId = ((warpId << 3) + b_colId) & (minorSize - 1);
    int factor_colId;
    int shared_colId;
    int shared_rowId;
    int out_colId;
    int out_rowId;
    

    // 申请共享内存
    __shared__ uint64_t shared_C[32 * 32];
    // 申请wmma片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

    int addrOff = (batchId*nttSize) +(unitId << log_unitSize);
    int inputIndex = addrOff + (tileId << log_tileSize) + in_colId+(b_rowId << log_minorSize);
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint64_t inputVal = input[inputIndex + (i << log_minorSize)];

        for (int j = 0; j < WORD_IN; j++)
        {
            b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
        }
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + i * MATRIX_OFFSET, WMMA_K);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint64_t inputVal = input[inputIndex + ((i + 16) << log_minorSize)];
        for (int j = 0; j < WORD_IN; j++)
        {
            b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
        }
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + (i + 8) * MATRIX_OFFSET, WMMA_K);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }
// 约简
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
        uint64_t result = Barrett64_gpu::reduction(a, modulus);
        shared_colId = c_rowId + ((frag_i >> 1) << 3);
        shared_rowId = (warpId << 3) + c_colId + (frag_i & 1);
        factor_colId = shared_rowId & (minorSize - 1);
        result = Barrett64_gpu::mult(result, factorTable[scale*factor_colId*shared_colId], modulus);
        shared_C[(shared_rowId<<5)+shared_colId] = result;
    }

    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint64_t inputVal = shared_C[((b_rowId + i) << 5) + (warpId << 3) + b_colId];

        for (int j = 0; j < WORD_IN; j++)
        {
            b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
        }
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }

    // 读取输入
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + i * MATRIX_OFFSET, WMMA_K);
    }

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint64_t inputVal = shared_C[((b_rowId + i + 16) << 5) + (warpId << 3) + b_colId];


        for (int j = 0; j < WORD_IN; j++)
        {
            b_frag[j].x[i] = (inputVal >> (8 * j)) & 0xFF;
        }
    }

    // 约简
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
        uint64_t result = Barrett64_gpu::reduction(a, modulus);
        shared_colId = (c_rowId + ((frag_i >> 1) << 3));
        tileId = shared_colId >> log_minorSize;
        out_rowId = (warpId << 3) + c_colId + (frag_i & 1);
        out_colId = shared_colId & (minorSize - 1);
        output[addrOff+(tileId<<log_tileSize)+(out_rowId<<log_minorSize)+out_colId] = result;
    }
}

__host__ void GPU_TensorNTT_7(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize)
{
    switch (log_nttSize)
    {
        case 12:
            tensorCore_7A<<<dim3(4, 1, batchSize), dim3(32, 4)>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(4, 1, batchSize), dim3(32, 4)>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 13:
            tensorCore_7A<<<dim3(8, 1, batchSize), dim3(32, 4)>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(8, 1, batchSize), dim3(32, 4)>>>(3, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 14:
            tensorCore_7A<<<dim3(16, 1, batchSize), dim3(32, 4)>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(16, 1, batchSize), dim3(32, 4)>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 15:
            tensorCore_7A<<<dim3(32, 1, batchSize), dim3(32, 4)>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(32, 1, batchSize), dim3(32, 4)>>>(5, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 16:
            tensorCore_7A<<<dim3(64, 1, batchSize), dim3(32, 4)>>>(6, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(64, 1, batchSize), dim3(32, 4)>>>(1, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 17:
            tensorCore_7A<<<dim3(128, 1, batchSize), dim3(32, 4)>>>(7, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(128, 1, batchSize), dim3(32, 4)>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 18:
            tensorCore_7A<<<dim3(256, 1, batchSize), dim3(32, 4)>>>(8, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(256, 1, batchSize), dim3(32, 4)>>>(3, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 19:
            tensorCore_7A<<<dim3(512, 1, batchSize), dim3(32, 4)>>>(9, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(512, 1, batchSize), dim3(32, 4)>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 20:
            tensorCore_7A<<<dim3(1024, 1, batchSize), dim3(32, 4)>>>(10, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(1024, 1, batchSize), dim3(32, 4)>>>(5, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 21:
            tensorCore_7A<<<dim3(2048, 1, batchSize), dim3(32, 4)>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7A<<<dim3(64, 32, batchSize), dim3(32, 4)>>>(6, log_nttSize, 5, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(2048, 1, batchSize), dim3(32, 4)>>>(1, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 22:
            tensorCore_7A<<<dim3(4096, 1, batchSize), dim3(32, 4)>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7A<<<dim3(128, 32, batchSize), dim3(32, 4)>>>(7, log_nttSize, 5, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(4096, 1, batchSize), dim3(32, 4)>>>(2, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 23:
            tensorCore_7A<<<dim3(8192, 1, batchSize), dim3(32, 4)>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7A<<<dim3(256, 32, batchSize), dim3(32, 4)>>>(9, log_nttSize, 10, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(8192, 1, batchSize), dim3(32, 4)>>>(3, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 24:
            tensorCore_7A<<<dim3(16384, 1, batchSize), dim3(32, 4)>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7A<<<dim3(512, 32, batchSize), dim3(32, 4)>>>(10, log_nttSize, 10, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_7B<<<dim3(16384, 1, batchSize), dim3(32, 4)>>>(4, input, output, majorMatrix, factorTable, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
    }
}
