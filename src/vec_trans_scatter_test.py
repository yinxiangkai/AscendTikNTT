import math
from functools import reduce as functools_reduce
import numpy as np
from tbe import tik
import tbe.common.platform as tbe_platform
from tbe.common.utils import para_check




DTYPE_SIZE = {
    'int8': 1,
    'float16': 2,
    'float32': 4,

}

data_type='int8'
dtype='int8'
k_size=64
n_size=64
k1=2
n=64
k0=32
tik_instance = tik.Tik(disable_debug=False)
kn_input_tensor = tik_instance.Tensor(data_type, (k_size, n_size), name="kn_input_tensor", scope=tik.scope_gm)
k1nk0_tensor = tik_instance.Tensor(data_type, (k_size // k0, n_size, k0), name="k1nk0_tensor",
                                          scope=tik.scope_gm)
with tik_instance.for_range(0, k1) as index:
        k1nk0_ub = tik_instance.Tensor(dtype, (n, k0), tik.scope_ubuf, "k1nk0_ub")
        src_ub = tik_instance.Tensor(dtype, (k0, n), tik.scope_ubuf, "src_ub")
        burst_len = k0 * n * DTYPE_SIZE[dtype] // 32
        tik_instance.data_move(src_ub, kn_input_tensor[index * k0 * n], 0, 1, burst_len, 0, 0)

        src_list_left = [src_ub[n * i] for i in range(16)]
        src_list_right = [src_ub[n * i+16*n_size] for i in range(16)]
        dst_list_up = [k1nk0_ub[32 * i] for i in range(16)]
        dst_list_down = [k1nk0_ub[32 * i +  16*k0] for i in range(16)]
        rep_times = n // k0
        dst_rep_stride = k0
        src_rep_stride = 1

        #左上 
        tik_instance.vec_trans_scatter(False, False, dst_list_up, src_list_left, rep_times, dst_rep_stride, src_rep_stride)
        #左下
        tik_instance.vec_trans_scatter(False, True, dst_list_down, src_list_left, rep_times, dst_rep_stride, src_rep_stride)
        # 右上
        tik_instance.vec_trans_scatter(True, False, dst_list_up, src_list_right, rep_times, dst_rep_stride, src_rep_stride)
        # 右下
        tik_instance.vec_trans_scatter(True, True, dst_list_down, src_list_right, rep_times, dst_rep_stride, src_rep_stride)

        tik_instance.data_move(k1nk0_tensor[index * k0 * n], k1nk0_ub, 0, 1, burst_len, 0, 0)



tik_instance.BuildCCE(kernel_name="vec_trans_scatter", inputs=[kn_input_tensor], outputs=[k1nk0_tensor])
rows, columns = 64, 64

# 定义矩阵的行数和列数
rows, columns = 64, 64

# 使用 arange 生成每行的基数，确保每行都是相同的数值
row_values = np.arange(1, rows + 1, dtype="int8").reshape(rows, 1)

# 使用广播填充整个矩阵
data_x = np.repeat(row_values, columns, axis=1)

feed_dict = {'kn_input_tensor': data_x}
model_data, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)

with open("model_data_output.txt", "w") as file:
    # 遍历每一个第一维度
    for i, matrix in enumerate(model_data):
        file.write(f"Tensor {i} (Matrix {i}):\n")
        
        # 遍历每一个第二维度
        for j, row in enumerate(matrix):
            # 将每一行写入，保持整齐的缩进
            file.write("  [" + ", ".join(map(str, row)) + "]\n")
        
        # 方便阅读，在每个矩阵之间添加空行
        file.write("\n")
