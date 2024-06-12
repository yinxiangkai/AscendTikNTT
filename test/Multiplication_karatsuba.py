import numpy as np
import math
import sys

def add(a, b):
    length = len(a)
    result = np.zeros(length, dtype=np.int32)
    for i in range(length):
        result[i] = int(a[i]) + int(b[i])
    return result

def subtract(a, b):
    length = len(a)
    result = np.zeros(length, dtype=np.int8)
    for i in range(length):
        value = int(a[i] - b[i])
        if value < 0:
            a[i + 1] -= 1
            value += 128
        result[i] = value & 0x7F
    return result

def karatsuba(a,b):
    print(a)
    if len(a) == 1:
        temp_tiling = np.zeros(2, dtype=np.int8)
        value = int(a[0]) * int(b[0])
        temp_tiling[0] = value & 0x7F
        temp_tiling[1] = (value >> 7) & 0x7F
        return temp_tiling
    
    if len(a)%2 != 0:
        a = np.pad(a , (0, 1), 'constant', constant_values=(0,))
        b = np.pad(b , (0, 1), 'constant', constant_values=(0,))

    half = len(a)//2
    
    # 分割向量
    low_a = a[half:]
    high_a = a[:half]
    low_b = b[half:]
    high_b = b[:half]

    z0 = karatsuba(low_a, low_b )
    z2 = karatsuba(high_a, high_b)
    z1 = karatsuba(add(low_a, high_a), add(low_b ,high_b))
    z1 = subtract(subtract(z1, z2), z0)

    temp_tiling= np.zeros(4*half, dtype=np.int8)

    # for i in range(len(z0)):
    #     temp_tiling[i] += z0[i]
    
    # for i in range(len(z1)):
    #     value = int(temp_tiling[i + half] + z1[i])
    #     temp_tiling[i + half] = value & 0x7F
    #     temp_tiling[i + half + 1] += (value >> 7)&0x1

    # for i in range(len(z2)):
    #     value = int(temp_tiling[i + half] + z2[i])
    #     temp_tiling[i + 2*half] = value & 0x7F
    #     temp_tiling[i + 2*half + 1] += (value >> 7)&0x1

    return temp_tiling



# 定义大整数
prime = 4179340454199820289
a = 1394649864822396625
b = 159035048546431000

# 计算标准模乘结果
result = (a * b) % prime
print("直接计算结果:", result)

# 分块
a_tiling = np.zeros(9, dtype=np.int8)
b_tiling = np.zeros(9, dtype=np.int8)
temp_tiling = np.zeros(18, dtype=np.int32)  # 结果需要足够的位数来存储中间结果
result_tiling = np.zeros(18, dtype=np.int8)  # 结果需要足够的位数来存储中间结果

# 切分块
for i in range(9):
    a_tiling[i] = (a >> (i * 7)) & 0x7F
    b_tiling[i] = (b >> (i * 7)) & 0x7F



# karatsuba乘法
karatsuba(a_tiling,b_tiling)

# 处理进位
for i in range(17):
    temp_tiling[i + 1] += temp_tiling[i] >> 7
    result_tiling[i] = temp_tiling[i]& 0x7F

# 重组结果
result_multiplied = 0
for i in range(18):
    result_multiplied += int(result_tiling[17 - i])
    if i != 17:
        result_multiplied <<= 7

# 取模
result_multiplied %= prime

# 打印结果
print("模拟乘法累加结果:", result_multiplied)



