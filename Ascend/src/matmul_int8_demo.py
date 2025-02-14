
import math
from functools import reduce as functools_reduce
import numpy as np
from tbe import tik
import tbe.common.platform as tbe_platform
from tbe.common.utils import para_check


def get_byte_len(dtype):
    index = 0
    for i in dtype:
        if i.isdigit():
            break
        index += 1
    return int(dtype[index:])//8 

if __name__ == "__main__":
    tik_instance = tik.Tik(disable_debug=False)
    m_size, k_size, n_size = 32, 64, 32
    m0, k0, n0 = 16, 32, 16
    dtype = "int8"
    out_dtype = "int32"
    # 申请全局变量空间
    mk_input_gm = tik_instance.Tensor(dtype, [32,64], name='mk_input_gm', scope=tik.scope_gm)
    kn_input_gm = tik_instance.Tensor(dtype, [64,32], name='kn_input_gm', scope=tik.scope_gm)
    mn_output_gm = tik_instance.Tensor(out_dtype, [32,32], name='mn_output_gm', scope=tik.scope_gm)
    mn_output_workspace = tik_instance.Tensor(out_dtype, [2,32,16], name='mn_output_l0c', scope=tik.scope_gm)

    
    # 申请cube输入输出缓存空间
    mk_input_l0a = tik_instance.Tensor(dtype, [2,32,32], name='mk_input_l0a', scope=tik.scope_cbuf)
    kn_input_l0b = tik_instance.Tensor(dtype, [2,32,32], name='kn_input_l0b', scope=tik.scope_cbuf)
    mn_output_l0c = tik_instance.Tensor(out_dtype, [2,32,16], name='mn_output_l0c', scope=tik.scope_cbuf_out)
    
    # 数据迁移与变形
    #!mk->k1mk0
    k1 = 2
    with tik_instance.for_range(0, k1) as i:
        tik_instance.data_move(mk_input_l0a[i,0,0], mk_input_gm[i*k0], 0, m_size, k0*1//32, (k_size-k0)*1//32, 0)


    #!kn->k1nk0
    k1 = 2
    with tik_instance.for_range(0, k1) as index:
        # 申请临时变量，因为vec_trans_scatter的输入是ub
        k1nk0_ub = tik_instance.Tensor(dtype, (n_size, k0), name="k1nk0_ub", scope=tik.scope_ubuf)
        src_ub = tik_instance.Tensor(dtype, (k0, n_size), name="src_ub", scope=tik.scope_ubuf)
        burst_len = k0 * n_size *get_byte_len(dtype) // 32
        tik_instance.data_move(src_ub, kn_input_gm[index * k0 * n_size], 0, 1, burst_len, 0, 0)

        src_list_left = [src_ub[n_size * i] for i in range(16)]
        src_list_right = [src_ub[n_size * i+16*n_size] for i in range(16)]
        dst_list_up = [k1nk0_ub[32 * i] for i in range(16)]
        dst_list_down = [k1nk0_ub[32 * i +  16*k0] for i in range(16)]
        rep_times = n_size // k0
        dst_rep_stride = k0
        src_rep_stride = 1

        #左上 
        tik_instance.vec_trans_scatter(False,True, dst_list_up, src_list_left, rep_times, dst_rep_stride, src_rep_stride)
        #左下
        tik_instance.vec_trans_scatter(False, False, dst_list_down, src_list_left, rep_times, dst_rep_stride, src_rep_stride)
        # 右上
        tik_instance.vec_trans_scatter(True, True, dst_list_up, src_list_right, rep_times, dst_rep_stride, src_rep_stride)
        # 右下
        tik_instance.vec_trans_scatter(True, False, dst_list_down, src_list_right, rep_times, dst_rep_stride, src_rep_stride)

        tik_instance.data_move(kn_input_l0b[index * k0 * n_size], k1nk0_ub, 0, 1, burst_len, 0, 0)
    
    
    # matmul计算
    tik_instance.matmul(mn_output_l0c, mk_input_l0a, kn_input_l0b, m_size, k_size, n_size)
    tik_instance.fixpipe(mn_output_gm, mn_output_l0c, 2, 64, 0, 0, extend_params={"relu": False})   

    tik_instance.BuildCCE(kernel_name="matmul_sample", inputs=[mk_input_gm,kn_input_gm], outputs=[mn_output_gm], config={"save_temp_cce_file": True})

    data_x = np.ones((32, 64)).astype("int8")
    data_y = np.ones((64, 32)).astype("int8")
    feed_dict = {'mk_input_gm': data_x, 'kn_input_gm': data_y}
    model_data, = tik_instance.tikdb.start_debug(feed_dict=feed_dict,interactive=True)
    with open("model_output.txt", "w") as file:
        for row in model_data:
            file.write("[" + ", ".join(map(str, row)) + "]\n")
            

