import math
import random
from sympy import isprime
import numpy as np

def bit_reverse_copy(a, n):
    result = [0] * n
    for i in range(n):
        rev_i = 0
        for j in range(int(math.log2(n))):
            rev_i <<= 1
            rev_i |= (i >> j) & 1
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
    length = 2
    while length <= n:
        wlen = modexp(root, n // length, mod)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length // 2] * w % mod
                a[i + j] = (u + v) % mod
                a[i + j + length // 2] = (u - v + mod) % mod
                w = w * wlen % mod
        length *= 2
    return a

def intt(a, n, mod, root):
    a = bit_reverse_copy(a, n)
    length = 2
    while length <= n:
        wlen = modexp(root, n // length, mod)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length // 2] * w % mod
                a[i + j] = (u + v) % mod
                a[i + j + length // 2] = (u - v + mod) % mod
                w = w * wlen % mod
        length *= 2
    inv_n = modexp(n, mod - 2, mod)
    print(inv_n)
    return [(x * inv_n) % mod for x in a]

# 示例调用
M = 4179340454199820289  # 模数
N = 1 << 20  # 规模
r = 1394649864822396625  # 原根
i_r = 159035048546431000  # 原根的逆
a = [0] * N
b = [0] * N
c = [0] * N

for i in range(N // 2):
    a[i] = 1
    b[i] = 1

ntt_a = ntt(a.copy(), N, M, r)

matrix = np.ones((1024, 1024), dtype=int)
for i in range(1024):
    for j in range(1024):
        matrix[i][j] =ntt_a[j+i*1024]

ntt_b = ntt(b.copy(), N, M, r)

for i in range(N):
    c[i] = ntt_a[i] * ntt_b[i] % M

ntt_c = intt(c.copy(), N, M, i_r)



    