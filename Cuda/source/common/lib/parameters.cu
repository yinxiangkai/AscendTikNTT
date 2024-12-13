#include "parameters.cuh"
#include <common.cuh>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sys/types.h>

TensorNTTParameters::TensorNTTParameters(int _logn, int _scheme)
{
    log_nttSize = _logn;
    nttSize = 1 << log_nttSize;
    scheme = _scheme;
    modularPool();
    unityRootPool();
    factorTableGenerator();

    subSizeGenerator();
    majorGenerator();
    minorGenerator();
}

TensorNTTParameters::TensorNTTParameters() {}

void TensorNTTParameters::modularPool()
{
    customAssert((4 <= log_nttSize) && (log_nttSize <= 28), "LOGN should be in range 4 to 28.");
    static uint64_t primes[] = {288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161,
                                288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161,
                                288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161,
                                288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161};
    Modulus prime(primes[log_nttSize - 4]);
    modulus = prime;
}

void TensorNTTParameters::unityRootPool()
{
    customAssert((4 <= log_nttSize) && (log_nttSize <= 28), "LOGN should be in range 4 to 28.");
    static uint64_t root[] = {280001509889547967, 210266531692667396, 92821654137585834,  260960605224279425, 162099094686911402, 107238753920285700, 32470950644356983,
                              117301231978048845, 216255723180573998, 250307369399164732, 219193197408848622, 149769509321831632, 249206447238919942, 120532064279427626,
                              268277439247089826, 199361202157402006, 197906187731194318, 67698184942246295,  214361899706311334, 123157478892883930, 43444860593900277,
                              63937158349600286,  276957054920883394, 279172744045218282, 225865349704673648};

    unityRoot = root[log_nttSize - 4];
}

void TensorNTTParameters::subSizeGenerator()
{
    switch (scheme)
    {
        case 0:
            log_majorSize = 4;
            log_minorSize = log_nttSize - log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = 1 << (log_minorSize * 2);
            break;
        case 1:
            log_majorSize = 4;
            log_minorSize = log_nttSize - log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = 1 << (log_minorSize * 2);
            break;
        case 2:
            log_majorSize = 4;
            log_minorSize = log_nttSize % log_majorSize == 0 ? 4 : log_nttSize % log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = majorMatrixSize;
            break;
        case 3:
            log_majorSize = 4;
            log_minorSize = log_nttSize % log_majorSize == 0 ? 4 : log_nttSize % log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = majorMatrixSize;
            break;
        case 4:
            log_majorSize = 4;
            log_minorSize = log_nttSize % log_majorSize == 0 ? 4 : log_nttSize % log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = majorMatrixSize;
            break;
        case 5:
            log_majorSize = 5;
            log_minorSize = log_nttSize % log_majorSize == 0 ? 5 : log_nttSize % log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = majorMatrixSize;
            break;
        case 6:
            log_majorSize = 5;
            log_minorSize = log_nttSize % log_majorSize == 0 ? 5 : log_nttSize % log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = majorMatrixSize;
            break;
        case 7:
            log_majorSize = 5;
            log_minorSize = log_nttSize % log_majorSize == 0 ? 5 : log_nttSize % log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = majorMatrixSize;
            break;
        case 8:
            log_majorSize = 5;
            log_minorSize = log_nttSize % log_majorSize == 0 ? 5 : log_nttSize % log_majorSize;
            majorSize = 1 << log_majorSize;
            minorSize = 1 << log_minorSize;
            majorMatrixSize = 1 << (log_majorSize * 2);
            minorMatrixSize = majorMatrixSize;
            break;
        default:
            throw std::runtime_error("Invalid choice.\n");
    }
}

void TensorNTTParameters::majorGenerator()
{
    uint64_t exp = int(nttSize / majorSize);
    uint64_t majorUnitRoot = Barrett64_cpu::exp(unityRoot, exp, modulus);
    int halfMatrixSize = majorMatrixSize / 2;
    int halfMajorSize = majorSize / 2;
    switch (scheme)
    {
        case 0:
            majorMatrix.resize(majorMatrixSize * 8, 0);
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[k * majorMatrixSize + i * majorSize + j] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 1:   // 2
            majorMatrix.resize(majorMatrixSize * 8, 0);
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[k * majorMatrixSize + i * majorSize + j] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 2:   // 3
            majorMatrix.resize(majorMatrixSize * 8, 0);
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[k * majorMatrixSize + i * majorSize + j] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 3:   // 3
            majorMatrix.resize(majorMatrixSize * 8, 0);
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[k * majorMatrixSize + i * majorSize + j] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 4:   // 3
            majorMatrix.resize(majorMatrixSize * 8, 0);
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[k * majorMatrixSize + i * majorSize + j] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 5:   // 6
            majorMatrix.resize(majorMatrixSize * 8, 0);
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[k * majorMatrixSize + i * 32 + j] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 6:   // 6
            majorMatrix.resize(majorMatrixSize * 8, 0);
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[k * majorMatrixSize + i * 32 + j] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 7:   // 5
            majorMatrix.resize(majorMatrixSize * 8, 0);
            int offset;
            int row;
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    if (j < halfMajorSize)
                    {
                        offset = 0;
                        row = j;
                    }
                    else
                    {
                        offset = 8;
                        row = (j - halfMajorSize);
                    }
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[(k + offset) * halfMatrixSize + i * halfMajorSize + row] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 8:   // 5
            majorMatrix.resize(majorMatrixSize * 8, 0);
            // int offset;
            // int row;
            for (int i = 0; i < majorSize; i++)
            {
                for (int j = 0; j < majorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(majorUnitRoot, tempEep, modulus);
                    if (j < halfMajorSize)
                    {
                        offset = 0;
                        row = j;
                    }
                    else
                    {
                        offset = 8;
                        row = (j - halfMajorSize);
                    }
                    for (int k = 0; k < 8; k++)
                    {
                        majorMatrix[(k + offset) * halfMatrixSize + i * halfMajorSize + row] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;

        default:
            throw std::runtime_error("Invalid choice.\n");
    }
}

void TensorNTTParameters::minorGenerator()
{
    uint64_t exp = int(nttSize / minorSize);
    uint64_t minorUnitRoot = Barrett64_cpu::exp(unityRoot, exp, modulus);
    int round = majorSize / minorSize;
    int halfMatrixSize = minorMatrixSize / 2;
    int offset;
    switch (scheme)
    {
        case 0:
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < minorSize; i++)
            {
                for (int j = 0; j < minorSize; j++)
                {
                    uint64_t tempEep = int(i * j);
                    uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                    int addOffset = ((i / 16) * (minorSize / 16) + (j / 16)) * 256 + (i % 16) * 16 + j % 16;
                    for (int k = 0; k < 8; k++)
                    {
                        minorMatrix[k * minorMatrixSize + addOffset] = uint8_t((temp >> (8 * k)) & 0xFF);
                    }
                }
            }
            break;
        case 1:   // 2
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < round; i++)
            {
                for (int m = 0; m < minorSize; m++)
                {
                    for (int n = 0; n < minorSize; n++)
                    {
                        uint64_t tempEep = int(m * n);
                        uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                        int addOffset = (i * minorSize + m) * majorSize + i * minorSize + n;
                        for (int k = 0; k < 8; k++)
                        {
                            minorMatrix[k * minorMatrixSize + addOffset] = uint8_t((temp >> (8 * k)) & 0xFF);
                        }
                    }
                }
            }
            break;
        case 2:   // 3
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < round; i++)
            {
                for (int m = 0; m < minorSize; m++)
                {
                    for (int n = 0; n < minorSize; n++)
                    {
                        uint64_t tempEep = int(m * n);
                        uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                        int addOffset = (i * minorSize + m) * majorSize + i * minorSize + n;
                        for (int k = 0; k < 8; k++)
                        {
                            minorMatrix[k * minorMatrixSize + addOffset] = uint8_t((temp >> (8 * k)) & 0xFF);
                        }
                    }
                }
            }
            break;
        case 3:   // 3
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < round; i++)
            {
                for (int m = 0; m < minorSize; m++)
                {
                    for (int n = 0; n < minorSize; n++)
                    {
                        uint64_t tempEep = int(m * n);
                        uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                        int addOffset = (i * minorSize + m) * majorSize + i * minorSize + n;
                        for (int k = 0; k < 8; k++)
                        {
                            minorMatrix[k * minorMatrixSize + addOffset] = uint8_t((temp >> (8 * k)) & 0xFF);
                        }
                    }
                }
            }
            break;
        case 4:   // 3
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < round; i++)
            {
                for (int m = 0; m < minorSize; m++)
                {
                    for (int n = 0; n < minorSize; n++)
                    {
                        uint64_t tempEep = int(m * n);
                        uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                        int addOffset = (i * minorSize + m) * majorSize + i * minorSize + n;
                        for (int k = 0; k < 8; k++)
                        {
                            minorMatrix[k * minorMatrixSize + addOffset] = uint8_t((temp >> (8 * k)) & 0xFF);
                        }
                    }
                }
            }
            break;

        case 5:   // 6
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < round; i++)
            {
                for (int m = 0; m < minorSize; m++)
                {
                    for (int n = 0; n < minorSize; n++)
                    {
                        uint64_t tempEep = int(m * n);
                        uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                        int colId = i * minorSize + n;
                        int rowId = i * minorSize + m;
                        for (int k = 0; k < 8; k++)
                        {
                            minorMatrix[k * minorMatrixSize + rowId * 32 + colId] = uint8_t((temp >> (8 * k)) & 0xFF);
                        }
                    }
                }
            }
            break;
        case 6:   // 6
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < round; i++)
            {
                for (int m = 0; m < minorSize; m++)
                {
                    for (int n = 0; n < minorSize; n++)
                    {
                        uint64_t tempEep = int(m * n);
                        uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                        int colId = i * minorSize + n;
                        int rowId = i * minorSize + m;
                        for (int k = 0; k < 8; k++)
                        {
                            minorMatrix[k * minorMatrixSize + rowId * 32 + colId] = uint8_t((temp >> (8 * k)) & 0xFF);
                        }
                    }
                }
            }
            break;
        case 7:   // 6
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < round; i++)
            {
                for (int m = 0; m < minorSize; m++)
                {
                    for (int n = 0; n < minorSize; n++)
                    {
                        uint64_t tempEep = int(m * n);
                        uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                        int colId = i * minorSize + n;
                        int rowId = i * minorSize + m;
                        for (int k = 0; k < 8; k++)
                        {
                            minorMatrix[k * minorMatrixSize + rowId * 32 + colId] = uint8_t((temp >> (8 * k)) & 0xFF);
                        }
                    }
                }
            }
            break;

        case 8:   // 5
            minorMatrix.resize(minorMatrixSize * 8);
            for (int i = 0; i < round; i++)
            {
                for (int m = 0; m < minorSize; m++)
                {
                    for (int n = 0; n < minorSize; n++)
                    {
                        uint64_t tempEep = int(m * n);
                        uint64_t temp = Barrett64_cpu::exp(minorUnitRoot, tempEep, modulus);
                        int colId = i * minorSize + n;
                        int rowId = i * minorSize + m;
                        if (colId < 16)
                        {
                            offset = 0;
                        }
                        else
                        {
                            offset = 8;
                            colId = (colId - 16);
                        }
                        for (int k = 0; k < 8; k++)
                        {
                            minorMatrix[(k + offset) * halfMatrixSize + rowId * 16 + colId] = uint8_t((temp >> (8 * k)) & 0xFF);
                        }
                    }
                }
            }
            break;
        default:
            throw std::runtime_error("Invalid choice.\n");
    }
}

void TensorNTTParameters::factorTableGenerator()
{
    factorTable.push_back(1);
    for (int i = 1; i < nttSize; i++)
    {
        factorTable.push_back(Barrett64_cpu::mult(factorTable[i - 1], unityRoot, modulus));
    }
}

void TensorNTTParameters::getSchemes() {}

TensorNTTParameters::~TensorNTTParameters()
{
    majorMatrix.clear();
    minorMatrix.clear();
    majorTable.clear();
    minorTable.clear();
    factorTable.clear();
}


/************************************************
 *                                               *
 *                 NTTParameters                 *
 *                                               *
 *************************************************/

NTTParameters::NTTParameters() {}
NTTParameters::NTTParameters(int LOGN)
{
    log_nttSize = LOGN;
    nttSize = 1 << log_nttSize;
    log_rootSize = log_nttSize - 1;
    rootSize = 1 << log_rootSize;
    modulus = modularPool();
    unityRoot = unityRootPool();
    inverseUnityRoot = Barrett64_cpu::inv(unityRoot, modulus);
    nttInv = Barrett64_cpu::inv(nttSize, modulus);
    unityRootTableGenerator();
    inverseUnityRootTableGenerator();
    unityRootReverseTableGenerator();
    inverseUnityRootReverseTableGenerator();
}

Modulus NTTParameters::modularPool()
{
    customAssert((4 <= log_nttSize) && (log_nttSize <= 28), "LOGN should be in range 4 to 28.");
    static uint64_t primes[] = {288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161,
                                288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161,
                                288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161,
                                288230385815388161, 288230385815388161, 288230385815388161, 288230385815388161};

    Modulus prime(primes[log_nttSize - 4]);
    return prime;
}

uint64_t NTTParameters::unityRootPool()
{
    customAssert((4 <= log_nttSize) && (log_nttSize <= 28), "LOGN should be in range 4 to 28.");
    static uint64_t root[] = {280001509889547967, 210266531692667396, 92821654137585834,  260960605224279425, 162099094686911402, 107238753920285700, 32470950644356983,
                              117301231978048845, 216255723180573998, 250307369399164732, 219193197408848622, 149769509321831632, 249206447238919942, 120532064279427626,
                              268277439247089826, 199361202157402006, 197906187731194318, 67698184942246295,  214361899706311334, 123157478892883930, 43444860593900277,
                              63937158349600286,  276957054920883394, 279172744045218282, 225865349704673648};
    return root[log_nttSize - 4];
}


void NTTParameters::unityRootTableGenerator()
{
    unityRootTable.push_back(1);
    for (int i = 1; i < rootSize; i++)
    {
        unityRootTable.push_back(Barrett64_cpu::mult(unityRootTable[i - 1], unityRoot, modulus));
    }
}

void NTTParameters::inverseUnityRootTableGenerator()
{
    inverseUnityRootTable.push_back(1);
    for (int i = 1; i < rootSize; i++)
    {
        inverseUnityRootTable.push_back(Barrett64_cpu::mult(inverseUnityRootTable[i - 1], inverseUnityRoot, modulus));
    }
}

void NTTParameters::unityRootReverseTableGenerator()
{
    unityRootReverseTable = reverseCopy(unityRootTable, log_rootSize);
}

void NTTParameters::inverseUnityRootReverseTableGenerator()
{
    inverseUnityRootReverseTable = reverseCopy(inverseUnityRootTable, log_rootSize);
}


int NTTParameters::bitreverse(int index, int bitwidth)
{
    int res = 0;
    for (int i = 0; i < bitwidth; i++)
    {
        res <<= 1;
        res |= (index >> i) & 1;
    }
    return res;
}

std::vector<uint64_t> NTTParameters::reverseCopy(std::vector<uint64_t> input, int bitwidth)
{
    std::vector<uint64_t> output;
    for (int i = 0; i < input.size(); i++)
    {
        output.push_back(input[bitreverse(i, bitwidth)]);
    }
    return output;
}