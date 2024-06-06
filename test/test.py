import math
import random
from sympy import isprime
import numpy as np


def bit_reverse_copy(a, n):
    result = [0] * n
    for i in range(n):
        rev_i = 0
        for j in range(20):
            rev_i<<=1
            rev_i |= (i>>j)&1
        result[rev_i] = a[i]
    return result

def modexp(a, b, mod):
    res = 1
    a = a % mod
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % mod
        b = b >> 1
        a = (a * a) % mod
    return res


def ntt(a, n, mod, root):
    a = bit_reverse_copy(a, n)
    print("位逆序: ", a[:10])
    for i in range(1, 20):
        length=1<<i
        wlen = modexp(root,  n// length, mod)
        for j in range(0, n, length):
            w = 1
            for k in range(length // 2):
                u = a[j+k]
                v = a[j+k + length // 2] * w % mod
                a[j+k] = (u + v) % mod
                a[j+k + length // 2] = (u - v) % mod
                w = w * wlen % mod
    return a


# 示例调用
M = 4179340454199820289  # 模数
N = 1<<20  # 规模
r = 1394649864822396625  # 原根
a = [1]*N  # 示例多项式，规模为 N

print("输入多项式: ", a[:10])  # 只打印前10个元素以免输出过长
ntt_result = ntt(a, N, M, r)
print("NTT变换结果: ", ntt_result[:10])  # 只打印前10个元素以免输出过长
  # 只打印前10个元素以免输出过长



