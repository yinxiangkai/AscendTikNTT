import numpy as np

# 定义参数
N = 2**20  # 输入规模
root = 3  # 原根
mod = 4179340454199820289  # 模数

# 计算逆元
def mod_inverse(a, mod):
    return pow(a, mod - 2, mod)

# 预计算幂
def precompute_powers(n, root, mod):
    powers = [1] * n
    for i in range(1, n):
        powers[i] = (powers[i-1] * root) % mod
    return powers

# 计算NTT
def ntt(a, n, root, mod):
    a = np.array(a, dtype=np.int64)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = pow(root, n // length, mod)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = (a[i + j + length // 2] * w) % mod
                a[i + j] = (u + v) % mod
                a[i + j + length // 2] = (u - v) % mod
                w = (w * wlen) % mod
        length *= 2
    return a

# 示例输入数据
data = [1] * N  # 用 1 填充的数据，仅作示例，可以替换为其他输入

# 计算NTT
ntt_result = ntt(data, N, root, mod)
print("NTT 结果:", ntt_result[:10])  # 仅打印前10个结果

