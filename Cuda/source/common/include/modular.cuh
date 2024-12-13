#pragma once
#include <cstdint>
#include <sys/types.h>

class Modulus {
  public:
    uint64_t value;   // 模数
    uint64_t bit;     // 模数的位数位宽
    uint64_t beta;    // beta=2^(2*bit)/value下取整

    Modulus(uint64_t _value)
    {
        value = _value;
        bit   = bit_generate();
        beta  = beta_generate();
    }
    Modulus()
    {
        value = 0;
        bit   = 0;
        beta  = 0;
    }

  private:
    uint64_t bit_generate()
    {
        return log2(value) + 1;
    }

    uint64_t beta_generate()
    {
        __uint128_t temp = (__uint128_t)(1) << ((2 * bit) + 1);
        temp             = temp / value;

        return temp;
    }
};



class Barrett64_cpu {
  public:
    // Modular Addition for 64 bit
    // result = (input1 + input2) % modulus
    static uint64_t add(uint64_t& input1, uint64_t& input2, Modulus& modulus)
    {
        uint64_t sum = input1 + input2;
        sum          = (sum >= modulus.value) ? (sum - modulus.value) : sum;

        return sum;
    }

    // Modular Substraction for 64 bit
    // result = (input1 - input2) % modulus
    static uint64_t sub(uint64_t& input1, uint64_t& input2, Modulus& modulus)
    {
        uint64_t dif = input1 + modulus.value;
        dif          = dif - input2;
        dif          = (dif >= modulus.value) ? (dif - modulus.value) : dif;

        return dif;
    }

    // Modular Multiplication for 64 bit
    // result = (input1 * input2) % modulus
    static uint64_t mult(uint64_t& input1, uint64_t& input2, Modulus& modulus)
    {
        __uint128_t mult = (__uint128_t)input1 * (__uint128_t)input2;

        __uint128_t r = mult >> (modulus.bit - 2);
        r             = r * (__uint128_t)modulus.beta;
        r             = r >> (modulus.bit + 3);
        r             = r * (__uint128_t)modulus.value;
        mult          = mult - r;

        uint64_t result = uint64_t(mult & UINT64_MAX);

        if (result >= modulus.value) {
            result -= modulus.value;
        }

        return result;
    }

    // Modular Exponentiation for 64 bit
    // result = (base ^ exponent) % modulus
    static uint64_t exp(uint64_t& base, uint64_t& exponent, Modulus& modulus)
    {
        // with window method
        uint64_t result = 1;

        int modulus_bit = log2(modulus.value) + 1;
        for (int i = modulus_bit - 1; i > -1; i--) {
            result = mult(result, result, modulus);
            if (((exponent >> i) & 1u)) {
                result = mult(result, base, modulus);
            }
        }

        return result;
    }

    // Modular Inversion for 64 bit
    // result = (1 / input) % modulus
    static uint64_t inv(uint64_t& input, Modulus& modulus)
    {
        uint64_t index = modulus.value - 2;
        return exp(input, index, modulus);
    }
};
class Barrett64_gpu {
  public:
    // Modular Addition for 64 bit
    // result = (input1 + input2) % modulus
    static __device__ __forceinline__ uint64_t add(uint64_t& input1, uint64_t& input2, Modulus& modulus)
    {
        uint64_t sum = input1 + input2;
        sum          = (sum >= modulus.value) ? (sum - modulus.value) : sum;

        return sum;
    }

    // Modular Substraction for 64 bit
    // result = (input1 - input2) % modulus
    static __device__ __forceinline__ uint64_t sub(uint64_t& input1, uint64_t& input2, Modulus& modulus)
    {
        uint64_t dif = input1 + modulus.value;
        dif          = dif - input2;
        dif          = (dif >= modulus.value) ? (dif - modulus.value) : dif;

        return dif;
    }

    // Modular Multiplication for 64 bit
    // result = (input1 * input2) % modulus
    static __device__ __forceinline__ uint64_t mult(uint64_t& input1, uint64_t& input2, Modulus& modulus)
    {
        __uint128_t mult = (__uint128_t)input1 * (__uint128_t)input2;

        __uint128_t r = mult >> (modulus.bit - 2);
        r             = r * (__uint128_t)modulus.beta;
        r             = r >> (modulus.bit + 3);
        r             = r * (__uint128_t)modulus.value;
        mult          = mult - r;

        uint64_t result = uint64_t(mult & UINT64_MAX);

        if (result >= modulus.value) {
            result -= modulus.value;
        }

        return result;
    }

    // Barrett Reduction for 64 bit
    // result = input1 % modulus
    static __device__ __forceinline__ uint64_t reduction(__uint128_t& input1, Modulus& modulus)
    {
        __uint128_t r = input1 >> (modulus.bit - 2);
        r             = r * (__uint128_t)modulus.beta;
        r             = r >> (modulus.bit + 3);
        r             = r * (__uint128_t)modulus.value;
        input1        = input1 - r;

        uint64_t result = uint64_t(input1 & UINT64_MAX);

        if (result >= modulus.value) {
            result -= modulus.value;
        }

        return result;
    }
};
