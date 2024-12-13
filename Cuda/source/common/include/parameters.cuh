#pragma once
#include "modular.cuh"
#include <cstdint>
#include <sys/types.h>
#include <vector>

class TensorNTTParameters {
  public:
    int log_nttSize;      // 输入规模以2为底的对数
    uint64_t nttSize;     // 输入规模
    Modulus modulus;      // 模数
    uint64_t unityRoot;   // 单位根

    int scheme;   // 方案

    int log_majorSize;    // 主要的NTT大小以2为底的对数
    int log_minorSize;    // 次要的NTT大小以2为底的对数
    uint64_t majorSize;   // 主要的NTT大小
    uint64_t minorSize;   // 次要的NTT大小
    uint64_t majorMatrixSize;
    uint64_t minorMatrixSize;   // 主要的NTT大小的逆

    std::vector<uint64_t> factorTable;
    std::vector<uint8_t> majorMatrix;
    std::vector<uint8_t> minorMatrix;
    std::vector<uint64_t> majorTable;
    std::vector<uint64_t> minorTable;

    TensorNTTParameters(int _logn, int _scheme);
    TensorNTTParameters();
    ~TensorNTTParameters();

  private:
    void modularPool();            // 模数池
    void unityRootPool();          // 单位根池
    void getSchemes();             // 方案信息
    void factorTableGenerator();   // 单位根表生成器
    void subSizeGenerator();       // 矩阵大小生成器
    void majorGenerator();         // 主要的NTT矩阵生成器
    void minorGenerator();         // 次要的NTT矩阵生成器
};

class NTTParameters {
  public:
    int log_nttSize;     // 输入规模以2为底的对数
    uint64_t nttSize;    // 输入规模
    Modulus modulus;     // 模数
    int log_rootSize;    // 单位根数量以2为底的对数
    uint64_t rootSize;   // 单位根数量

    uint64_t unityRoot;          // 单位根
    uint64_t inverseUnityRoot;   // 单位根的逆
    uint64_t nttInv;             // 输入规模的逆

    std::vector<uint64_t> unityRootTable;
    std::vector<uint64_t> inverseUnityRootTable;
    std::vector<uint64_t> unityRootReverseTable;
    std::vector<uint64_t> inverseUnityRootReverseTable;

    NTTParameters(int LOGN);

    NTTParameters();
    std::vector<uint64_t> unityRootTableGenerator_GPU(std::vector<uint64_t> table);
    int bitreverse(int index, int bitwidth);
    std::vector<uint64_t> reverseCopy(std::vector<uint64_t> input, int bitwidth);

  private:
    Modulus modularPool();
    uint64_t unityRootPool();
    void unityRootTableGenerator();
    void inverseUnityRootTableGenerator();
    void unityRootReverseTableGenerator();
    void inverseUnityRootReverseTableGenerator();
};