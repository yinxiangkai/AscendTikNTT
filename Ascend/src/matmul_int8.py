
import math
from functools import reduce as functools_reduce
import numpy as np
from tbe import tik
import tbe.common.platform as tbe_platform
from tbe.common.utils import para_check

# 递归函数将多维数组转换为带有 `[]` 的字符串
def tensor_to_string(arr):
    if arr.ndim == 1:
        # 一维数组，转为 `[item1, item2, ...]`
        return '[' + ', '.join(map(str, arr)) + ']'
    else:
        # 对更高维的数组进行递归
        return '[' + ',\n '.join(tensor_to_string(subarr) for subarr in arr) + ']'


def mk_to_k1mk0(tik_instance, matrix_a_gm, matrix_a_workspace_gm, dtype,dtype_size, k1_size, m_size, k_size,  k0_size, m_tiling_size, m_cycle_times, m_thread_num, k_tiling_size, k_cycle_times, k_thread_num):
    # 申请UB空间
    with tik_instance.for_range(0, m_cycle_times, thread_num = m_thread_num) as m_cycle:
        with tik_instance.for_range(0, k_cycle_times, thread_num = k_thread_num) as k_cycle:
            temp_ub = tik_instance.Tensor(dtype, [m_tiling_size,k_tiling_size], name = 'temp_ub', scope = tik.scope_ubuf)
            data_move = tik_instance.data_move(temp_ub, matrix_a_gm[m_cycle * m_tiling_size, k_cycle * k_tiling_size], 0, m_tiling_size, k_tiling_size* dtype_size // 32, (k_size - k_tiling_size)*dtype_size // 32, 0)
           
            with tik_instance.for_range(0, k1_size) as k1:
                tik_instance.data_move(matrix_a_workspace_gm[m_cycle, k_cycle, k1, 0, 0], temp_ub[k1 * k0_size], 0, m_tiling_size, k0_size * dtype_size // 32, (k_tiling_size - k0_size)*dtype_size // 32, 0)

def kn_to_k1nk0(tik_instance, matrix_b_gm, matrix_b_workspace_gm, dtype,dtype_size, k1_size, n_size, k_size,  k0_size, n_tiling_size, n_cycle_times, n_thread_num, k_tiling_size, k_cycle_times, k_thread_num):
    # 申请UB空间
    with tik_instance.for_range(0, k_cycle_times, thread_num = k_thread_num) as k_cycle:
        with tik_instance.for_range(0, n_cycle_times, thread_num = n_thread_num) as n_cycle:
            with tik_instance.for_range(0, k1_size) as k1:      
                temp_k0n_ub = tik_instance.Tensor(dtype, (k0_size, n_tiling_size), tik.scope_ubuf, "temp_k0n_ub")
                temp_nk0_ub = tik_instance.Tensor(dtype, (n_tiling_size, k0_size), tik.scope_ubuf, "temp_nk0_ub")
                data_move = tik_instance.data_move(temp_k0n_ub, matrix_b_gm[k_cycle * k_tiling_size+k0_size*k1, n_cycle * n_tiling_size], 0, k0_size, n_tiling_size*dtype_size//32, (n_size-n_tiling_size)*dtype_size//32, 0)

                src_list_left = [temp_k0n_ub[n_tiling_size * i] for i in range(16)]
                src_list_right = [temp_k0n_ub[n_tiling_size * i+16*n_tiling_size] for i in range(16)]
                dst_list_up = [temp_nk0_ub[32 * i] for i in range(16)]
                dst_list_down = [temp_nk0_ub[32 * i +  16*k0_size] for i in range(16)]
                rep_times = n_tiling_size // k0_size
                dst_rep_stride = k0_size
                src_rep_stride = 1

                #左上 
                tik_instance.vec_trans_scatter(False, False, dst_list_up, src_list_left, rep_times, dst_rep_stride, src_rep_stride)
                #左下
                tik_instance.vec_trans_scatter(False, True, dst_list_down, src_list_left, rep_times, dst_rep_stride, src_rep_stride)
                # 右上
                tik_instance.vec_trans_scatter(True, False, dst_list_up, src_list_right, rep_times, dst_rep_stride, src_rep_stride)
                # 右下
                tik_instance.vec_trans_scatter(True, True, dst_list_down, src_list_right, rep_times, dst_rep_stride, src_rep_stride)

                tik_instance.data_move(matrix_b_workspace_gm[k_cycle, n_cycle, k1, 0, 0], temp_nk0_ub, 0, 1, k0_size * n_tiling_size * dtype_size // 32, 0, 0)



def matrix_mul(tik_instance, params):
    # 获取输入参数
    m_size = params['M']
    k_size = params['K']
    n_size = params['N']
    dtype = params['dtype']
    dtype_size = params['dtype_size']
    out_dtype = params['out_dtype']
    out_dtype_size = params['out_dtype_size']
    m_tiling_size = params['m_tiling_size']
    m_cycle_times = params['m_cycle_times']
    m_thread_num = params['m_thread_num']
    k_tiling_size = params['k_tiling_size']
    k_cycle_times = params['k_cycle_times']
    k_thread_num = params['k_thread_num']
    n_tiling_size = params['n_tiling_size']
    n_cycle_times = params['n_cycle_times']
    n_thread_num = params['n_thread_num']
    m0_size = 16
    k0_size = 32
    n0_size = 16
    k1_size = k_tiling_size // k0_size
    n1_size = n_tiling_size // n0_size
    block_size = 32

    # 申请全局变量空间
    matrix_a_gm = tik_instance.Tensor(dtype, [m_size, k_size], name = 'matrix_a_gm', scope = tik.scope_gm)
    matrix_b_gm = tik_instance.Tensor(dtype, [k_size, n_size], name = 'matrix_b_gm', scope = tik.scope_gm)
    matrix_c_gm = tik_instance.Tensor(out_dtype, [m_size, n_size], name = 'matrix_c_gm', scope = tik.scope_gm)

    # 申请L1空间
    kmk0_l1a = tik_instance.Tensor(dtype, [k1_size, m_tiling_size, k0_size], name = 'kmk0_l1a', scope = tik.scope_cbuf)
    knk0_l1b = tik_instance.Tensor(dtype, [k1_size, n_tiling_size, k0_size], name = 'knk0_l1b', scope = tik.scope_cbuf)
    # 申请L0空间
    n1mn0_l1c = tik_instance.Tensor(out_dtype, [n1_size, n_tiling_size,n0_size], name = 'mnk0_l1c', scope = tik.scope_cbuf_out)
   
    # 申请工作空间
    matrix_a_workspace_gm = tik_instance.Tensor(dtype, [m_cycle_times, k_cycle_times, k1_size, m_tiling_size, k0_size], name = 'matrix_a_workspace', scope = tik.scope_gm, is_workspace=True)
    matrix_b_workspace_gm = tik_instance.Tensor(dtype, [k_cycle_times, n_cycle_times, k1_size, n_tiling_size, k0_size], name = 'matrix_b_workspace', scope = tik.scope_gm, is_workspace=True)

    mn_ub = tik_instance.Tensor(out_dtype, [m_tiling_size, n_tiling_size], name = 'mn_ub', scope = tik.scope_ubuf)
    temp_mn_gm = tik_instance.Tensor(out_dtype, [m_tiling_size,n_tiling_size], name = 'temp_mn_gm', scope = tik.scope_gm,is_workspace=True)
    # 数据搬运与格式转换
    mk_to_k1mk0(tik_instance, matrix_a_gm, matrix_a_workspace_gm, dtype,dtype_size, k1_size, m_size, k_size,  k0_size, m_tiling_size, m_cycle_times, m_thread_num, k_tiling_size, k_cycle_times, k_thread_num)
    kn_to_k1nk0(tik_instance, matrix_b_gm, matrix_b_workspace_gm, dtype,dtype_size, k1_size, n_size, k_size,  k0_size, n_tiling_size, n_cycle_times, n_thread_num, k_tiling_size, k_cycle_times, k_thread_num)



    # 矩阵乘法计算 
    with tik_instance.for_range(0, m_cycle_times, thread_num = m_thread_num) as m_cycle:
        with tik_instance.for_range(0, n_cycle_times, thread_num = n_thread_num) as n_cycle:
            with tik_instance.for_range(0, k_cycle_times, thread_num = k_thread_num) as k_cycle:
                tik_instance.data_move(kmk0_l1a, matrix_a_workspace_gm[m_cycle, k_cycle, 0, 0, 0], 0, 1, m_tiling_size * k_tiling_size * dtype_size // 32, 0, 0)
                tik_instance.data_move(knk0_l1b, matrix_b_workspace_gm[k_cycle, n_cycle, 0, 0, 0], 0, 1, n_tiling_size * k_tiling_size * dtype_size // 32, 0, 0)
                with tik_instance.if_scope(k_cycle == 0):
                    tik_instance.matmul(n1mn0_l1c, kmk0_l1a, knk0_l1b, m_tiling_size, k_tiling_size, n_tiling_size, True)
                with tik_instance.else_scope():
                    tik_instance.matmul(n1mn0_l1c, kmk0_l1a, knk0_l1b, m_tiling_size, k_tiling_size, n_tiling_size, False)

            tik_instance.fixpipe(temp_mn_gm, n1mn0_l1c, n1_size, n0_size*m_tiling_size*out_dtype_size//32, 0, 0)
            tik_instance.data_move(mn_ub, temp_mn_gm, 0, 1, m_tiling_size * n_tiling_size * out_dtype_size // 32, 0, 0)
            with tik_instance.for_range(0, n1_size) as n1:
                tik_instance.data_move(matrix_c_gm[m_cycle * m_tiling_size, n_cycle * n_tiling_size+n1*n0_size ], mn_ub, 0, m_tiling_size, n0_size * out_dtype_size // 32, 0, (n_size - n0_size) * out_dtype_size // 32)


    tik_instance.BuildCCE(kernel_name="matrix_mul", inputs=[matrix_a_gm,matrix_b_gm], outputs=[matrix_c_gm], config={"save_temp_cce_file": True})



if __name__ == "__main__":
    M=64
    K=128
    N=64
    # 输入参数和tiling信息
    params = {
        'M': M,
        'K': K,
        'N': N,
        'dtype': "int8",
        'dtype_size': 1,
        'out_dtype': "int32",
        'out_dtype_size': 4,
        'm_tiling_size': 32,
        'm_cycle_times': 2,
        'm_thread_num': 1,
        'k_tiling_size': 64,
        'k_cycle_times': 2,
        'k_thread_num': 1,
        'n_tiling_size': 32,
        'n_cycle_times': 2,
        'n_thread_num': 1,   
    }

    tik_instance = tik.Tik(disable_debug=False)
    matrix_mul(tik_instance, params)

    # testc
    matrix_a = np.ones((M, K)).astype("int8")
    matrix_b = np.ones((K, N)).astype("int8")
    # cycle_length = 64
    # row_cycle = np.arange(1, cycle_length + 1).astype("int8")  # 生成从1到64的数组

    # # 确保行向量足够长以填充整个矩阵列
    
    # matrix_a = np.tile(row_cycle, (M, (K // cycle_length) + 1))[:, :K]
    
    # row_values = np.arange(1, K + 1, dtype="int8").reshape(K, 1)
    # matrix_b = np.repeat(row_values, N, axis=1)
    # matrix_b = np.tile(row_cycle, (K, (N // cycle_length) + 1))[:, :N]

    # 将序列扩展到合适的大小
    feed_dict = {'matrix_a_gm': matrix_a, 'matrix_b_gm': matrix_b}
    matrix_c, = tik_instance.tikdb.start_debug(feed_dict=feed_dict,interactive=True)
    # with open("tensor.txt", "w") as file:
    #     for row in matrix_c:
    #         file.write("[" + ", ".join(map(str, row)) + "]\n") 
     
    tensor_str = tensor_to_string(matrix_c)

    # 保存到 txt 文件中
    with open('tensor.txt', 'w') as f:
        f.write(tensor_str)