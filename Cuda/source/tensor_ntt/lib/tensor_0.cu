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

__global__ void tensorCore_0(const uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, uint64_t* factorTable, Modulus modulus, int nttSize)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int subId = blockIdx.x;
    int nttId = blockIdx.y;

    int warpNum = blockDim.y;
    int subNum = gridDim.x;
    int rowSize = 16 * subNum * warpNum;

    int rowId = laneId % 16;
    int lineId = laneId / 16;
    int colId = 16 * (warpNum * subId + warpId) + lineId * 8;
    int inputIdx = nttId * nttSize + rowId * rowSize + colId;
    int outputIdx = nttId * nttSize + colId * 16 + rowId;

    // 申请共享内存
    extern __shared__ uint64_t shared_mem[];
    auto shared_B = reinterpret_cast<uint8_t(*)[WORD_IN][WMMA_M * WMMA_K]>(shared_mem);

    // 读取输入
    for (int i = 0; i < FRAG_NUM; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            shared_B[warpId][j][rowId * WMMA_K + lineId * 8 + i] = uint8_t((input[inputIdx + i] >> (j * 8)) & 0xff);
        }
    }

    // 申请矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

    // 初始化矩阵片段
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    for (int i = 0; i < WORD_IN; i++)
    {
        wmma::load_matrix_sync(a_frag[i], majorMatrix + 256 * i, WMMA_K);
        wmma::load_matrix_sync(b_frag[i], shared_B[warpId][i], WMMA_N);
    }

    // 计算
    for (int i = 0; i < WORD_IN; i++)
    {
        for (int j = 0; j < WORD_IN; j++)
        {
            wmma::mma_sync(c_frag[i + j], a_frag[i], b_frag[j], c_frag[i + j]);
        }
    }
    auto shared_C = reinterpret_cast<int32_t(*)[2][WMMA_M * WMMA_N]>(shared_mem);

    // 约简
    for (int frag_i = 0; frag_i < FRAG_NUM; ++frag_i)
    {
        // 将计算结果拼接。
        uint64_t concate55[WORD_OUT / 4];
        for (int j = 0; j < WORD_OUT / 4; ++j)
        {
            concate55[j] = c_frag[j * 4 + 0].x[frag_i];
            concate55[j] += static_cast<uint64_t>(c_frag[j * 4 + 1].x[frag_i]) << 8;
            concate55[j] += static_cast<uint64_t>(c_frag[j * 4 + 2].x[frag_i]) << 16;
            concate55[j] += static_cast<uint64_t>(c_frag[j * 4 + 3].x[frag_i]) << 24;
        }
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
        uint64_t temp = shared_C[warpId][0][rowId * 16 + lineId * 8 + i];
        temp += ((uint64_t)shared_C[warpId][1][rowId * 16 + lineId * 8 + i]) << 32;
        temp = Barrett64_gpu::mult(temp, factorTable[(colId + i) * rowId], modulus);
        output[outputIdx + i * 16] = temp;
    }
}


__global__ void tensorCore_0(const uint64_t* input, uint64_t* output, const uint8_t* minorMatrix, Modulus modulus, int nttSize)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int subId = blockIdx.x;
    int nttId = blockIdx.y;

    int warpNum = blockDim.y;
    int subNum = gridDim.x;
    int minorSize = subNum * warpNum * 16;
    int matrixSize = minorSize * minorSize;
    int tileSize = warpNum * 256;

    int colId = laneId / 16;
    int lineId = laneId % 16;

    int rowIndex = 16 * (subId * warpNum + warpId) + lineId;
    int sharedIndex = lineId * 16 + colId * 8;
    int inputIndex = nttId * nttSize + warpId * 256 + sharedIndex;
    int outputIndex = nttId * nttSize + rowIndex * 16 + colId * 8;
    int matrixOffset = minorSize * (subId * warpNum + warpId) * 16;

    // 申请共享内存
    extern __shared__ uint64_t shared_mem[];
    auto shared_B = reinterpret_cast<uint8_t(*)[WORD_IN][WMMA_M * WMMA_K]>(shared_mem);

    // 申请矩阵片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag[WORD_IN];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> b_frag[WORD_IN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[WORD_OUT];

    // 初始化矩阵片段
    for (int i = 0; i < WORD_OUT; i++)
    {
        wmma::fill_fragment(c_frag[i], 0);
    }

    for (int i = 0; i < subNum; i++)
    {

        // 读取输入
        for (int j = 0; j < FRAG_NUM; j++)
        {
            uint64_t value = input[inputIndex + i * tileSize + j];
            for (int k = 0; k < WORD_IN; k++)
            {
                shared_B[warpId][k][sharedIndex + j] = (uint8_t)((value >> (k * 8)) & 0xff);
            }
        }

        // 计算
        for (int j = 0; j < warpNum; j++)
        {

            for (int k = 0; k < WORD_IN; k++)
            {
                wmma::load_matrix_sync(a_frag[k], minorMatrix + k * matrixSize + matrixOffset + i * tileSize + j * 256, WMMA_K);
                wmma::load_matrix_sync(b_frag[k], shared_B[j][k], WMMA_N);
            }

            for (int k = 0; k < WORD_IN; k++)
            {
                for (int l = 0; l < WORD_IN; l++)
                {
                    wmma::mma_sync(c_frag[k + l], a_frag[k], b_frag[l], c_frag[k + l]);
                }
            }
        }
    }
    auto shared_C = reinterpret_cast<int32_t(*)[2][WMMA_M * WMMA_N]>(shared_mem);
    // 数据处理
    for (int frag_i = 0; frag_i < 8; ++frag_i)
    {
        // 将计算结果拼接。
        uint64_t concate55[WORD_OUT / 4];
        for (int j = 0; j < WORD_OUT / 4; ++j)
        {
            concate55[j] = c_frag[j * 4 + 0].x[frag_i];
            concate55[j] += static_cast<uint64_t>(c_frag[j * 4 + 1].x[frag_i]) << 8;
            concate55[j] += static_cast<uint64_t>(c_frag[j * 4 + 2].x[frag_i]) << 16;
            concate55[j] += static_cast<uint64_t>(c_frag[j * 4 + 3].x[frag_i]) << 24;
        }
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
        c_frag[0].x[frag_i] = uint32_t(temp & 0xffffffff);
        c_frag[1].x[frag_i] = uint32_t((temp >> 32) & 0xffffffff);
        // 写回
    }
    wmma::store_matrix_sync(shared_C[warpId][0], c_frag[0], WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(shared_C[warpId][1], c_frag[1], WMMA_N, wmma::mem_row_major);

    for (int i = 0; i < FRAG_NUM; i++)
    {
        uint64_t temp = shared_C[warpId][0][lineId * 16 + colId * 8 + i];
        temp += ((uint64_t)shared_C[warpId][1][lineId * 16 + colId * 8 + i]) << 32;
        output[outputIndex + i] = temp;
    }
}

__host__ void GPU_TensorNTT_0(uint64_t* input, uint64_t* output, const uint8_t* majorMatrix, const uint8_t* minorMatrix, uint64_t* factorTable, int log_nttSize, Modulus modulus,
                              int batchSize, int nttSize)
{

    switch (log_nttSize)
    {
        case 8:
            tensorCore_0<<<dim3(1, batchSize), dim3(32, 1), 256 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(1, batchSize), dim3(32, 1), 256 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 9:
            tensorCore_0<<<dim3(1, batchSize), dim3(32, 2), 512 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(1, batchSize), dim3(32, 2), 512 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 10:
            tensorCore_0<<<dim3(1, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(1, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 11:
            tensorCore_0<<<dim3(2, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(2, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 12:
            tensorCore_0<<<dim3(4, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(4, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 13:
            tensorCore_0<<<dim3(8, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(8, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 14:
            tensorCore_0<<<dim3(16, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(16, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 15:
            tensorCore_0<<<dim3(32, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(32, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 16:
            tensorCore_0<<<dim3(64, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(64, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 17:
            tensorCore_0<<<dim3(128, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(128, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 18:
            tensorCore_0<<<dim3(256, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(256, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 19:
            tensorCore_0<<<dim3(512, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(512, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 20:
            tensorCore_0<<<dim3(1024, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(1024, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 21:
            tensorCore_0<<<dim3(2048, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(2048, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 22:
            tensorCore_0<<<dim3(4096, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(4096, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 23:
            tensorCore_0<<<dim3(8192, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(8192, batchSize), dim3(32, 4), 1024 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
        case 24:
            tensorCore_0<<<dim3(8192, batchSize), dim3(32, 8), 2048 * sizeof(uint64_t)>>>(input, input, majorMatrix, factorTable, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            tensorCore_0<<<dim3(8192, batchSize), dim3(32, 8), 2048 * sizeof(uint64_t)>>>(input, output, minorMatrix, modulus, nttSize);
            CUDA_CHECK(cudaGetLastError());
            break;
    }
}
