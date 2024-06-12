import numpy as np
import math

def get_beta(p, k, b):
    beta = b ** k
    s = 0
    while beta > s:
        s = beta
        beta = 2 * beta - math.floor((p * math.floor((beta ** 2)>>(k))) >>( k))
    t = b ** (2 * k) - p * beta

    while t < 0:
        beta -= 1
        t += p

    return beta

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

# 乘法实现
for i in range(9):
    for j in range(9):
        temp_tiling[i + j] += int(a_tiling[i]) * int(b_tiling[j])

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


print("模拟乘法累加结果:", result_multiplied%prime)


# 巴列特取模操作
length =math.ceil(math.log2(prime))
beta=get_beta(prime, length, 2)

t=result_multiplied>>(length-1)
t=t*beta
t=t>>(length+1)
z1= result_multiplied & ((1<<(length+1))-1)
z2= (t * prime) & ((1<<(length+1))-1)
z=z1-z2
if z<0:
    z+=(1<<(length+1))
while(z>=prime):
    z-=prime
    
print("巴列特取模结果",z)

