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

__global__ void __launch_bounds__(512)
    tensorCore_4A(int logStep, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int blockId = blockIdx.x;
    int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int warpNum = blockDim.y;
    int blockNum = gridDim.x;
    int subNum = gridDim.y;
    // int unitId = blockId * warpNum + warpId;

    int blockStep = (blockNum * warpNum) << 1;
    int subSize = blockStep << 4;
    int scale = subNum;

    int factor_rowId = laneId & 0xf;
    int c_rowId = laneId >> 2;
    // int c_colId;
    int lineId = laneId & 3;

    int factor_colId;

    // 申请矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_IN];
    // 申请共享内存
    __shared__ uint64_t shared_temp[16][32][4];

    if (logStep == 0)
    {
        int addrOff = (batchId << log_nttSize) + subId * subSize;
        const uint8_t* inputIndex = reinterpret_cast<const uint8_t*>(input + addrOff + (blockId << 5) + (warpId << 1));
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + i * WMMA_M * WMMA_K, WMMA_K);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }

        wmma::load_matrix_sync(b_frag, inputIndex, blockStep << 3);

#pragma unroll
        for (int i = 0; i < WORD_IN; i++)
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
            shared_temp[warpId][c_rowId + ((i & 1) << 3) + ((i >> 1) << 4)][lineId] = Barrett64_gpu::reduction(a, modulus);
        }
        __uint128_t a = (__uint128_t)shared_temp[warpId][laneId][0] + ((__uint128_t)shared_temp[warpId][laneId][1] << 16) + ((__uint128_t)shared_temp[warpId][laneId][2] << 32) +
                        ((__uint128_t)shared_temp[warpId][laneId][3] << 48);

        uint64_t result = Barrett64_gpu::reduction(a, modulus);
        factor_colId = (blockId << 5) + (warpId << 1) + (laneId >> 4);
        result = Barrett64_gpu::mult(result, factorTable[scale * factor_rowId * factor_colId], modulus);
        output[addrOff + factor_rowId * blockStep + factor_colId] = result;
    }
    else
    {
        __shared__ uint64_t shared_C[32 * 16];

        uint8_t* C_shared8 = reinterpret_cast<uint8_t*>(shared_C);
        int addrOff = (batchId << log_nttSize) + subId * subSize;
        const uint8_t* inputIndex = reinterpret_cast<const uint8_t*>(input + addrOff + (blockId << 1));
        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::load_matrix_sync(a_frag[i], majorMatrix + i * WMMA_M * WMMA_K, WMMA_K);
        }

        for (int i = 0; i < WORD_IN; i++)
        {
            wmma::fill_fragment(c_frag[i], 0);
        }
        wmma::load_matrix_sync(b_frag, inputIndex + (warpId << (logStep + 3)), blockStep << 3);

#pragma unroll
        for (int i = 0; i < WORD_IN; i++)
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
            shared_temp[warpId][c_rowId + ((i & 1) << 3) + ((i >> 1) << 4)][lineId] = Barrett64_gpu::reduction(a, modulus);
        }
        __uint128_t a = (__uint128_t)shared_temp[warpId][laneId][0] + ((__uint128_t)shared_temp[warpId][laneId][1] << 16) + ((__uint128_t)shared_temp[warpId][laneId][2] << 32) +
                        ((__uint128_t)shared_temp[warpId][laneId][3] << 48);

        uint64_t result = Barrett64_gpu::reduction(a, modulus);

        factor_colId = (blockId << 1) + (laneId >> 4) + (warpId << logStep);
        result = Barrett64_gpu::mult(result, factorTable[scale * factor_rowId * factor_colId], modulus);
        shared_C[(factor_rowId << 1) + (laneId >> 4) + (warpId << 5)] = result;

        scale = scale << 4;
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::fill_fragment(c_frag[j], 0);
        }
        wmma::load_matrix_sync(b_frag, C_shared8 + (warpId << 4), 256);

#pragma unroll
        for (int i = 0; i < WORD_IN; i++)
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
            shared_temp[warpId][c_rowId + ((i & 1) << 3) + ((i >> 1) << 4)][lineId] = Barrett64_gpu::reduction(a, modulus);
        }
        a = (__uint128_t)shared_temp[warpId][laneId][0] + ((__uint128_t)shared_temp[warpId][laneId][1] << 16) + ((__uint128_t)shared_temp[warpId][laneId][2] << 32) +
            ((__uint128_t)shared_temp[warpId][laneId][3] << 48);

        result = Barrett64_gpu::reduction(a, modulus);
        factor_colId = blockId + (laneId >> 4);
        // result = Barrett64_gpu::mult(result, factorTable[scale * warpId * factor_colId], modulus);
        output[addrOff + (blockId << 1) + (laneId >> 4) + (factor_rowId << logStep) + warpId * blockStep] = result;
        // output[blockId*32*16+warpId*32+laneId] = result;
    }
}

__global__ void tensorCore_4B(int log_minorSize, int log_nttSize, const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable,
                              const uint8_t* minorMatrix, Modulus modulus)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int blockId = blockIdx.x;
    // int subId = blockIdx.y;
    int batchId = blockIdx.z;

    int warpNum = blockDim.y;
    int unitId = blockId * warpNum + warpId;

    int log_unitSize = 8;
    int log_tileSize = log_minorSize + 4;
    int scale = 1 << (log_nttSize - 4 - log_minorSize);
    int addOff = (batchId << log_nttSize) + (unitId << log_unitSize);

    int b_colId = laneId >> 2;
    int b_rowId = (laneId & 3) << 2;
    int c_colId = ((laneId & 3) << 1);
    int c_rowId = laneId >> 2;
    int factor_colId;
    int factor_rowId;
    int sharedIndex = (b_rowId << 4) + b_colId;

    int tileHalfNum = FRAG_NUM >> log_minorSize;   // 是否占据整个unit
    int tileWholeFlag = 1 - (log_minorSize >> 2);
    int minorMod = (1 << log_minorSize) - 1;
    int in_tileId = b_colId >> log_minorSize;
    int in_colId = b_colId & minorMod;

    int inputIndex = addOff + (b_rowId << log_minorSize) + in_colId + (in_tileId << log_tileSize);

    // 申请矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

    extern __shared__ uint64_t shared_mem[];
    auto shared_C = reinterpret_cast<int32_t(*)[2][WMMA_M * WMMA_K]>(shared_mem);

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        int addTemp = ((i * tileHalfNum) << log_tileSize) + ((i >> tileWholeFlag) << 3);
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            uint64_t inputVal = input[inputIndex + addTemp + (j << log_minorSize)];

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
        wmma::load_matrix_sync(a_frag[i], majorMatrix + i * WMMA_M * WMMA_K, WMMA_K);
    }
#pragma unroll
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

        factor_colId = (c_colId + ((frag_i >> 2) << 3) + (frag_i & 1)) & minorMod;
        factor_rowId = c_rowId + (((frag_i >> 1) & 1) << 3);
        // 旋转因子
        temp = Barrett64_gpu::mult(temp, factorTable[factor_rowId * factor_colId * scale], modulus);
        // 写回
        c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
        c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
    }

    wmma::store_matrix_sync(shared_C[warpId][0], c_frag[0], WMMA_N, wmma::mem_col_major);
    wmma::store_matrix_sync(shared_C[warpId][1], c_frag[1], WMMA_N, wmma::mem_col_major);

/*******************************************
 *第二轮计算
 *********************************************/
#pragma unroll
    for (int m = 0; m < 2; m++)
    {
        for (int i = 0; i < 2; i++)
        {
            int addTemp = (i << 3);
            for (int j = 0; j < 4; j++)
            {
                uint64_t inputVal = shared_C[warpId][m][sharedIndex + (j << 4) + addTemp];
                for (int n = 0; n < 4; n++)
                {
                    b_frag[(m << 2) + n].x[i * 4 + j] = (inputVal >> (8 * n)) & 0xFF;
                }
            }
        }
    }

    // 数据填充
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], minorMatrix + i * WMMA_M * WMMA_K, WMMA_K);
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
        factor_colId = (c_rowId + (((frag_i >> 1) & 1) << 3));
        factor_rowId = c_colId + ((frag_i >> 2) << 3) + (frag_i & 1);
        output[addOff + (factor_rowId << log_minorSize) + ((factor_colId >> log_minorSize) << log_tileSize) + (factor_colId & minorMod)] = temp;
    }
}




__host__ void GPU_TensorNTT_4(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize, cudaStream_t stream)
{
    switch (log_nttSize)
    {
        case 12:
            tensorCore_4A<<<dim3(8, 1, batchSize), dim3(32, 16)>>>(0, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(4, 1, batchSize), dim3(32, 4), 1 << 13>>>(4, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 13:
            tensorCore_4A<<<dim3(16, 1, batchSize), dim3(32, 16)>>>(5, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(8, 1, batchSize), dim3(32, 4), 1 << 13>>>(1, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 14:
            tensorCore_4A<<<dim3(32, 1, batchSize), dim3(32, 16)>>>(6, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(16, 1, batchSize), dim3(32, 4), 1 << 13>>>(2, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 15:
            tensorCore_4A<<<dim3(64, 1, batchSize), dim3(32, 16)>>>(7, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(32, 1, batchSize), dim3(32, 4), 1 << 13>>>(3, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 16:
            tensorCore_4A<<<dim3(128, 1, batchSize), dim3(32, 16)>>>(8, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(64, 1, batchSize), dim3(32, 4), 1 << 13>>>(4, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 17:
            tensorCore_4A<<<dim3(256, 1, batchSize), dim3(32, 16)>>>(9, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4A<<<dim3(1, 256, batchSize), dim3(32, 16)>>>(0, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(128, 1, batchSize), dim3(32, 4), 1 << 13>>>(1, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 18:
            tensorCore_4A<<<dim3(512, 1, batchSize), dim3(32, 16)>>>(10, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4A<<<dim3(2, 256, batchSize), dim3(32, 16)>>>(0, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(256, 1, batchSize), dim3(32, 4), 1 << 13>>>(2, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 19:
            tensorCore_4A<<<dim3(1024, 1, batchSize), dim3(32, 16)>>>(11, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4A<<<dim3(4, 256, batchSize), dim3(32, 16)>>>(0, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(512, 1, batchSize), dim3(32, 4), 1 << 13>>>(3, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 20:
            tensorCore_4A<<<dim3(2048, 1, batchSize), dim3(32, 16)>>>(12, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4A<<<dim3(8, 256, batchSize), dim3(32, 16)>>>(0, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(1024, 1, batchSize), dim3(32, 4), 1 << 13>>>(4, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 21:
            tensorCore_4A<<<dim3(4096, 1, batchSize), dim3(32, 16)>>>(13, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4A<<<dim3(16, 256, batchSize), dim3(32, 16)>>>(5, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(2048, 1, batchSize), dim3(32, 4), 1 << 13>>>(1, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 22:
            tensorCore_4A<<<dim3(8192, 1, batchSize), dim3(32, 16)>>>(14, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4A<<<dim3(32, 256, batchSize), dim3(32, 16)>>>(6, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(4096, 1, batchSize), dim3(32, 4), 1 << 13>>>(2, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 23:
            tensorCore_4A<<<dim3(16384, 1, batchSize), dim3(32, 16)>>>(15, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4A<<<dim3(64, 256, batchSize), dim3(32, 16)>>>(7, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(8192, 1, batchSize), dim3(32, 4), 1 << 13>>>(3, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 24:
            tensorCore_4A<<<dim3(32768, 1, batchSize), dim3(32, 16)>>>(16, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4A<<<dim3(128, 256, batchSize), dim3(32, 16)>>>(8, log_nttSize, input, input, majorMatrix, factorTable, modulus);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_4B<<<dim3(16384, 1, batchSize), dim3(32, 4), 1 << 13>>>(4, log_nttSize, input, output, majorMatrix, factorTable, minorMatrix, modulus);
            CUDA_CHECK(cudaGetLastError());
            break;
    }
}
