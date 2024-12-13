import numpy as np

def modexp(a, b, mod):
    res = 1
    a = a % mod
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % mod
        b = b >> 1
        a = (a * a) % mod
    return res

# 定义参数
prime = 4179340454199820289  # 模数
omega = 1394649864822396625  # 原根
inv_omega = 159035048546431000  # 原根的逆
inv_size = 4179336468470169601  # 大小的逆
size = 1 << 20  # 数据大小
matrix_size = 1 << 10  # 矩阵大小
scale = size // matrix_size

# 生成数据,通过测试
matrixA = np.zeros((matrix_size, matrix_size), dtype=int)

for i in range(matrix_size):
    for j in range(matrix_size):
        if i*matrix_size+j < size//2:
            matrixA[i][j] = 1

#生成幂次表，通过测试
omega_powers = np.ones(size, dtype=int)
inv_omega_powers = np.ones(size, dtype=int)
for i in range(1, size):
    omega_powers[i] = int (omega_powers[i-1]) * int (omega) % prime
    inv_omega_powers[i] =int ( inv_omega_powers[i-1] )* int (inv_omega )% prime

# 构造运算矩阵,测试通过
matrixNTT = np.zeros((matrix_size, matrix_size), dtype=int)
inv_matrixNTT = np.zeros((matrix_size, matrix_size), dtype=int)
for i in range(matrix_size):
    for j in range(matrix_size):
        matrixNTT[i][j] = omega_powers[i*j*scale % size]
        inv_matrixNTT[i][j] = inv_omega_powers[i*j*scale % size]

#构造旋转因子矩阵
factor = np.zeros((matrix_size, matrix_size), dtype=int)
inv_factor = np.zeros((matrix_size, matrix_size), dtype=int)
for i in range(matrix_size):
    for j in range(matrix_size):
        factor[i][j] = omega_powers[i*j % size]
        inv_factor[i][j] = inv_omega_powers[i*j % size]

# NTT变换
#列变换
print("列变换")
reuslt = np.zeros((matrix_size, matrix_size), dtype=int)
for i in range(matrix_size):
    for j in range(matrix_size):
       for k in range(matrix_size):
            temp =  int(matrixNTT[i][k])*int(matrixA[k][j]) % prime
            reuslt[i][j] = int(int(reuslt[i][j]) + int(temp)) % prime


# 乘旋转因子
print("乘旋转因子")
for i in range(matrix_size):
    for j in range(matrix_size):
        matrixA[i][j] = int(reuslt[i][j])*int(factor[i][j]) % prime
           
#行变换
print("行变换")
matrixA = matrixA.T
reuslt1 = np.zeros((matrix_size, matrix_size), dtype=int)
for i in range(matrix_size):
    for j in range(matrix_size):
       for k in range(matrix_size):
            temp =  int(matrixNTT[i][k])*int(matrixA[k][j]) % prime
            reuslt1[i][j] = int(int(reuslt1[i][j]) + int(temp)) % prime

print(reuslt1)
