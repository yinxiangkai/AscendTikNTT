#include "merge_ntt_cpu.cuh"
#include "parameters.cuh"
#include <cstdint>

std::vector<uint64_t> schoolbook_poly_multiplication(std::vector<uint64_t> a, std::vector<uint64_t> b, Modulus modulus)
{
    int length = a.size() + b.size();
    std::vector<uint64_t> result_vector(length, 0);
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b.size(); j++) {
            uint64_t mult_result = Barrett64_cpu::mult(a[i], b[j], modulus);
            result_vector[i + j] = Barrett64_cpu::add(result_vector[i + j], mult_result, modulus);
        }
    }

    return result_vector;
}

MergeNTT::MergeNTT(NTTParameters parameters_)
{
    parameters = parameters_;
}

std::vector<uint64_t> MergeNTT::mult(std::vector<uint64_t>& input1, std::vector<uint64_t>& input2)
{
    std::vector<uint64_t> output;
    for (int i = 0; i < parameters.nttSize; i++) {
        output.push_back(Barrett64_cpu::mult(input1[i], input2[i], parameters.modulus));
    }

    return output;
}

// std::vector<uint64_t> MergeNTT::ntt(std::vector<uint64_t> &input)
// {
//     int n = parameters.nttSize;
//     for(int i = parameters.log_nttSize; i >0; i--)
//     {
//         int m = 1<<i;
//         uint64_t scale = n/m;
//         for(int k = 0; k <n ; k+=m)
//         {
//             for(int j = 0; j < m/2; j++)
//             {
//                 uint64_t u = input[j+k];
//                 uint64_t v = input[j+k+m/2];
//                 input[j+k] = Barrett64_cpu::add(u, v, parameters.modulus);
//                 input[j+k+m/2] = Barrett64_cpu::sub(u, v, parameters.modulus);
//                 input[j+k+m/2] = Barrett64_cpu::mult(input[j+k+m/2], parameters.unityRootTable[j*scale],
//                 parameters.modulus);
//             }
//         }
//     }

//     std::vector<uint64_t> output = reverseCopy(input);
//     return output;
// }

// std::vector<uint64_t> MergeNTT::intt(std::vector<uint64_t> &input)
// {
//     std::vector<uint64_t> output = reverseCopy(input);
//     int n = parameters.nttSize;
//     for(int i = 0; i < parameters.log_nttSize; i++)
//     {
//         int m = 1<<(i+1);
//         uint64_t scale = n/m;
//         for(int k= 0; k <n ; k+=m)
//         {
//             uint64_t omegaPower = 1;
//             for(int j = 0; j < m/2; j++)
//             {
//                 uint64_t u = output[j+k];
//                 uint64_t v = Barrett64_cpu::mult(output[j+k+m/2], parameters.inverseUnityRootTable[j*scale],
//                 parameters.modulus); output[j+k] = Barrett64_cpu::add(u, v, parameters.modulus); output[j+k+m/2] =
//                 Barrett64_cpu::sub(u, v, parameters.modulus);
//             }
//         }
//     }

//     for(int i = 0; i < n; i++)
//     {
//         output[i] = Barrett64_cpu::mult(output[i], parameters.nttInv, parameters.modulus);
//     }
//     return output;
// }




std::vector<uint64_t> MergeNTT::ntt(std::vector<uint64_t>& input)
{

    std::vector<uint64_t> output = input;

    int t = parameters.nttSize;
    int m = 1;

    while (m < parameters.nttSize) {
        t = t >> 1;

        for (int i = 0; i < m; i++) {
            int j1 = 2 * i * t;
            int j2 = j1 + t;

            uint64_t S = parameters.unityRootReverseTable[i];

            for (int j = j1; j < j2; j++) {
                uint64_t U = output[j];
                uint64_t V = Barrett64_cpu::mult(output[j + t], S, parameters.modulus);

                output[j]     = Barrett64_cpu::add(U, V, parameters.modulus);
                output[j + t] = Barrett64_cpu::sub(U, V, parameters.modulus);
            }
        }

        m = m << 1;
    }
    return output;
}

std::vector<uint64_t> MergeNTT::intt(std::vector<uint64_t>& input)
{
    std::vector<uint64_t> output = input;

    int t = 1;
    int m = parameters.nttSize;
    while (m > 1) {
        int j1 = 0;
        int h  = m >> 1;
        for (int i = 0; i < h; i++) {
            int j2 = j1 + t;

            uint64_t S = parameters.inverseUnityRootReverseTable[i];

            for (int j = j1; j < j2; j++) {
                uint64_t U = output[j];
                uint64_t V = output[j + t];

                output[j]     = Barrett64_cpu::add(U, V, parameters.modulus);
                output[j + t] = Barrett64_cpu::sub(U, V, parameters.modulus);
                output[j + t] = Barrett64_cpu::mult(output[j + t], S, parameters.modulus);
            }

            j1 = j1 + (t << 1);
        }

        t = t << 1;
        m = m >> 1;
    }

    for (int i = 0; i < parameters.nttSize; i++) {
        output[i] = Barrett64_cpu::mult(output[i], parameters.nttInv, parameters.modulus);
    }

    return output;
}
