#pragma once
#include <parameters.cuh>

std::vector<uint64_t> schoolbook_poly_multiplication(std::vector<uint64_t> a, std::vector<uint64_t> b, Modulus modulus);

class MergeNTT {
  public:
    NTTParameters parameters;

    MergeNTT(NTTParameters parameters_);

  public:
    std::vector<uint64_t> mult(std::vector<uint64_t>& input1, std::vector<uint64_t>& input2);

    std::vector<uint64_t> ntt(std::vector<uint64_t>& input);

    std::vector<uint64_t> intt(std::vector<uint64_t>& input);
};
