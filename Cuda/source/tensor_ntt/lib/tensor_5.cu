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
#define WMMA_M    16
#define WMMA_N    16
#define WMMA_K    16
#define WORD_IN   8
#define WORD_OUT  16
#define FRAG_NUM  8
#define COL_SIZE  32

__global__ void tensorCore_5A(int logStep, int log_nttSize, int log_subNum, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              Modulus modulus)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int unitId = threadIdx.z;
    int blockId = (blockIdx.x * blockDim.z + unitId);
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int warpInRow = warpId >> 1;
    int warpInCol = warpId & 1;

    int log_subSize = log_nttSize - log_subNum;
    int log_rowSize = log_subSize - 5;


    int scale = gridDim.y;

    int b_colId = (laneId >> 2) + (warpInCol << 4);
    int b_rowId = (laneId & 3) << 2;
    int c_colId = ((laneId & 3) << 1) + (warpInCol << 4);
    int c_rowId = (laneId >> 2) + (warpInRow << 4);

    // int out_colId = (warpInRow << 4) + c_rowId;
    // int out_rowId = (warpInCol << 4) + c_colId;
    int factor_colId;
    int factor_rowId;

    int sharedOff = (b_rowId << 5) + b_colId;
    int addOff = (batchId << log_nttSize) + (subId << log_subSize);
    // 申请共享内存
    extern __shared__ uint64_t shared_mem[];
    auto shared_C = reinterpret_cast<int32_t(*)[2][1024]>(shared_mem);
    // 申请wmma片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];


    if (logStep == 0)
    {
        int in_colId = (blockId << 5) + b_colId;
        int inputIdx = addOff + (b_rowId << log_rowSize) + in_colId;

#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                uint64_t inputVal = input[inputIdx + (j << log_rowSize) + (i << 3)];

                for (int k = 0; k < WORD_IN; k++)
                {
                    b_frag[k].x[i * 4 + j] = (inputVal >> (8 * k)) & 0xFF;
                }
            }
        }
        // for (int frag_i = 0; frag_i < 8; frag_i++)
        // {
        //     output[unitId * 1024 + warpId * 256 + laneId * 8 + frag_i] = log_rowSize;
        // }


        __syncthreads();
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + (i << 10) + (warpInRow << 9), COL_SIZE);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + (i << 10) + (warpInRow << 9) + WMMA_K, COL_SIZE);
        }

        inputIdx += 16 << log_rowSize;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                uint64_t inputVal = input[inputIdx + (j << log_rowSize) + (i << 3)];

                for (int k = 0; k < WORD_IN; k++)
                {
                    b_frag[k].x[i * 4 + j] = (inputVal >> (8 * k)) & 0xFF;
                }
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
            uint64_t temp = Barrett64_gpu::reduction(a, modulus);
            factor_colId = (blockId << 5) + c_colId + ((frag_i >> 2) << 3) + (frag_i & 1);
            factor_rowId = c_rowId + (((frag_i >> 1) & 1) << 3);
            temp = Barrett64_gpu::mult(temp, factorTable[factor_rowId * factor_colId * scale], modulus);
            // 写回
            output[addOff + (factor_rowId << log_rowSize) + factor_colId] = temp;
        }
    }
    else
    {
        int in_colId = blockId + (b_colId << logStep);
        int inputIdx = addOff + (b_rowId << log_rowSize) + in_colId;
// 读取输入
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            int addTemp = (i << 3) << logStep;
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                uint64_t inputVal = input[inputIdx + (j << log_rowSize) + addTemp];

                for (int k = 0; k < WORD_IN; k++)
                {
                    b_frag[k].x[i * 4 + j] = (inputVal >> (8 * k)) & 0xFF;
                }
            }
        }
        // for (int frag_i = 0; frag_i < 8; frag_i++)
        // {
        //     output[unitId * 1024 + warpId * 256 + laneId * 8 + frag_i] = b_frag[0].x[frag_i];
        // }
        __syncthreads();
        for (int i = 0; i < WORD_OUT; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + (i << 10) + (warpInRow << 9), COL_SIZE);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + (i << 10) + (warpInRow << 9) + WMMA_K, COL_SIZE);
        }

        inputIdx += 16 << log_rowSize;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            int addTemp = (i << 3) << logStep;
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                uint64_t inputVal = input[inputIdx + (j << log_rowSize) + addTemp];

                for (int k = 0; k < WORD_IN; k++)
                {
                    b_frag[k].x[i * 4 + j] = (inputVal >> (8 * k)) & 0xFF;
                }
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
            uint64_t temp = Barrett64_gpu::reduction(a, modulus);
            factor_colId = (blockId << 5) + c_colId + ((frag_i >> 2) << 3) + (frag_i & 1);
            factor_rowId = c_rowId + (((frag_i >> 1) & 1) << 3);
            temp = Barrett64_gpu::mult(temp, factorTable[factor_rowId * factor_colId * scale], modulus);
            // 写回
            c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
            c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
        }

        wmma::store_matrix_sync(shared_C[unitId][0] + (warpInCol << 9) + (warpInRow << 4), c_frag[0], COL_SIZE, wmma::mem_col_major);
        wmma::store_matrix_sync(shared_C[unitId][1] + (warpInCol << 9) + (warpInRow << 4), c_frag[1], COL_SIZE, wmma::mem_col_major);
        __syncthreads();

        scale = scale << 5;
        // 读取输入

        for (int i = 0; i < WORD_OUT; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + (i << 10) + (warpInRow << 9), COL_SIZE);
        }
#pragma unroll
        for (int m = 0; m < 2; m++)
        {
            for (int i = 0; i < 2; i++)
            {
                int addTemp = (i << 3);
                for (int j = 0; j < 4; j++)
                {
                    uint64_t inputVal = shared_C[unitId][m][sharedOff + (j << 5) + addTemp];
                    for (int n = 0; n < 4; n++)
                    {
                        b_frag[(m << 2) + n].x[i * 4 + j] = (inputVal >> (8 * n)) & 0xFF;
                    }
                }
            }
        }


        for (int i = 0; i < WORD_IN; i++)
        {
            for (int j = 0; j < WORD_IN; j++)
            {
                wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
            }
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + (i << 10) + (warpInRow << 9) + WMMA_K, COL_SIZE);
        }

        sharedOff += 512;
#pragma unroll
        for (int m = 0; m < 2; m++)
        {
            for (int i = 0; i < 2; i++)
            {
                int addTemp = (i << 3);
                for (int j = 0; j < 4; j++)
                {
                    uint64_t inputVal = shared_C[unitId][m][sharedOff + (j << 5) + addTemp];
                    for (int n = 0; n < 4; n++)
                    {
                        b_frag[(m << 2) + n].x[i * 4 + j] = (inputVal >> (8 * n)) & 0xFF;
                    }
                }
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
            uint64_t temp = Barrett64_gpu::reduction(a, modulus);

            // 写回
            factor_colId = ((c_rowId + (((frag_i >> 1) & 1) << 3)) << logStep) + blockId;
            factor_rowId = c_colId + ((frag_i >> 2) << 3) + (frag_i & 1);
            temp = Barrett64_gpu::mult(temp, factorTable[factor_rowId * blockId * scale], modulus);
            output[addOff + (factor_rowId << log_rowSize) + factor_colId] = temp;
            // output[unitId*1024+warpId*256+laneId*8+frag_i] = temp;
        }
    }
}

__global__ void tensorCore_5B(int log_minorSize, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              const uint8_t* minorMatrix, Modulus modulus)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int unitId = threadIdx.z;
    // int blockId = blockIdx.x;
    int batchId = blockIdx.y;

    int minorSize = 1 << log_minorSize;
    int log_tileSize = log_minorSize + 5;

    int warpInRow = warpId >> 1;
    int warpInCol = warpId & 1;

    int log_unitSize = 10;
    int scale = log_nttSize - log_tileSize;
    int minorMod = minorSize - 1;


    int b_colId = (laneId >> 2) + (warpInCol << 4);
    int b_rowId = (laneId & 3) << 2;
    int c_colId = ((laneId & 3) << 1) + (warpInCol << 4);
    int c_rowId = (laneId >> 2) + (warpInRow << 4);
    int in_colId;
    // int in_rowId;
    int out_colId = (warpInRow << 4) + c_rowId;
    int out_rowId = (warpInCol << 4) + c_colId;
    int factor_colId;
    int factor_rowId;

    int in_tileId;
    int tile_colId;
    int addOff = (batchId << log_nttSize) + (unitId << log_unitSize);
    int sharedOff = (b_rowId << 5) + b_colId;
    int inputIdx = addOff + (b_rowId << log_minorSize);

    // 申请共享内存
    extern __shared__ uint64_t shared_mem[];
    auto shared_C = reinterpret_cast<int32_t(*)[2][1024]>(shared_mem);

    // 申请wmma片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];


#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        in_colId = b_colId + (i << 3);
        in_tileId = in_colId >> log_minorSize;
        tile_colId = in_colId & minorMod;
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            uint64_t inputVal = input[inputIdx + (j << log_minorSize) + (in_tileId << log_tileSize) + tile_colId];

            for (int k = 0; k < WORD_IN; k++)
            {
                b_frag[k].x[i * 4 + j] = (inputVal >> (8 * k)) & 0xFF;
            }
        }
    }

    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + (i << 10) + (warpInRow << 9), COL_SIZE);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + (i << 10) + (warpInRow << 9) + WMMA_K, COL_SIZE);
    }

    inputIdx += 16 << log_minorSize;
#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        in_colId = b_colId + (i << 3);
        in_tileId = in_colId >> log_minorSize;
        tile_colId = in_colId & minorMod;
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            uint64_t inputVal = input[inputIdx + (j << log_minorSize) + (in_tileId << log_tileSize) + tile_colId];

            for (int k = 0; k < WORD_IN; k++)
            {
                b_frag[k].x[i * 4 + j] = (inputVal >> (8 * k)) & 0xFF;
            }
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
        uint64_t temp = Barrett64_gpu::reduction(a, modulus);
        factor_colId = (c_colId + ((frag_i >> 2) << 3) + (frag_i & 1)) & minorMod;
        factor_rowId = c_rowId + (((frag_i >> 1) & 1) << 3);
        temp = Barrett64_gpu::mult(temp, factorTable[factor_rowId * factor_colId * scale], modulus);
        // 写回
        c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
        c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
        // output[unitId * 1024 + warpId * 256 + laneId * 8 + frag_i] = temp;
    }
    wmma::store_matrix_sync(shared_C[unitId][0] + (warpInCol << 9) + (warpInRow << 4), c_frag[0], COL_SIZE, wmma::mem_col_major);
    wmma::store_matrix_sync(shared_C[unitId][1] + (warpInCol << 9) + (warpInRow << 4), c_frag[1], COL_SIZE, wmma::mem_col_major);
    __syncthreads();
    // 旋转因子与再分配

#pragma unroll
    for (int m = 0; m < 2; m++)
    {
        for (int i = 0; i < 2; i++)
        {
            int addTemp = (i << 3);
            for (int j = 0; j < 4; j++)
            {
                uint64_t inputVal = shared_C[unitId][m][sharedOff + (j << 5) + addTemp];
                for (int n = 0; n < 4; n++)
                {
                    b_frag[(m << 2) + n].x[i * 4 + j] = (inputVal >> (8 * n)) & 0xFF;
                }
            }
        }
    }
    // for (int frag_i = 0; frag_i < 8; frag_i++)
    // {
    //      output[unitId * 1024 + warpId * 256 + laneId * 8 + frag_i] = b_frag[1].x[frag_i];
    // }

    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }


    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], minorMatrix + (i << 10) + (warpInRow << 9), COL_SIZE);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }
    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], minorMatrix + (i << 10) + (warpInRow << 9) + WMMA_K, COL_SIZE);
    }

    sharedOff += 16 << 5;
#pragma unroll
    for (int m = 0; m < 2; m++)
    {
        for (int i = 0; i < 2; i++)
        {
            int addTemp = (i << 3);
            for (int j = 0; j < 4; j++)
            {
                uint64_t inputVal = shared_C[unitId][m][sharedOff + (j << 5) + addTemp];
                for (int n = 0; n < 4; n++)
                {
                    b_frag[(m << 2) + n].x[i * 4 + j] = (inputVal >> (8 * n)) & 0xFF;
                }
            }
        }
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
        factor_colId = out_colId + (((frag_i >> 1) & 1) << 3);
        factor_rowId = out_rowId + ((frag_i >> 2) << 3) + (frag_i & 1);
        // output[unitId*1024+warpId*256+laneId*8+frag_i] = temp;
        output[addOff + (factor_rowId << log_minorSize) + ((factor_colId >> log_minorSize) << log_tileSize) + (factor_colId & minorMod)] = temp;
    }
}

__host__ void GPU_TensorNTT_5(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize)
{
    switch (log_nttSize)
    {
        case 12:
            tensorCore_5A<<<dim3(4, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(4, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(2, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 13:
            tensorCore_5A<<<dim3(8, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(8, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(3, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 14:
            tensorCore_5A<<<dim3(16, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(16, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(4, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 15:
            tensorCore_5A<<<dim3(32, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(32, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(5, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 16:
            tensorCore_5A<<<dim3(64, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(6, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(64, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(1, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 17:
            tensorCore_5A<<<dim3(128, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(7, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(128, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(2, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 18:
            tensorCore_5A<<<dim3(256, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(8, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(256, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(3, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 19:
            tensorCore_5A<<<dim3(512, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(9, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(512, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(4, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 20:
            tensorCore_5A<<<dim3(1024, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(10, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(1024, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(5, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 21:
            tensorCore_5A<<<dim3(2048, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5A<<<dim3(64, 32, batchSize), dim3(32, 4, 1), 1 << 13>>>(11, log_nttSize, 5, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(2048, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(1, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 22:
            tensorCore_5A<<<dim3(4096, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(0, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5A<<<dim3(128, 32, batchSize), dim3(32, 4, 1), 1 << 13>>>(6, log_nttSize, 5, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(4096, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(2, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 23:
            tensorCore_5A<<<dim3(8192, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(13, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5A<<<dim3(8, 1024, batchSize), dim3(32, 4, 1), 1 << 13>>>(0, log_nttSize, 10, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(8192, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(3, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 24:
            tensorCore_5A<<<dim3(16384, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(14, log_nttSize, 0, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5A<<<dim3(16, 1024, batchSize), dim3(32, 4, 1), 1 << 13>>>(0, log_nttSize, 10, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_5B<<<dim3(16384, 1, batchSize), dim3(32, 4, 1), 1 << 13>>>(4, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
    }
}
