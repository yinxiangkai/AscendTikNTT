import math
from functools import reduce as functools_reduce
import numpy as np
import te as te
from te import tik
from te import platform as cce
from topi.cce import util


#ccec -c -O2 ../matmul_mod.cce --cce-aicore-arch=dav-m100 --cce-aicore-only -o mmatmul.o

def matmul_sample():
    tik_instance = tik.Tik()
    # 定义tensor
    a_gm = tik_instance.Tensor("float16", [8 * 1024*8 * 16], name='a_gm', scope=tik.scope_gm)
    b_gm = tik_instance.Tensor("float16", [8, 240, 16], name='b_gm', scope=tik.scope_gm)
    dst_gm = tik_instance.Tensor("float32", [16 * 1024*8 * 16], name='dst_gm', scope=tik.scope_gm)

    a_l1 = tik_instance.Tensor("float16", [8, 256, 16], name='a_l1', scope=tik.scope_cbuf)
    b_l1 = tik_instance.Tensor("float16", [8, 240, 16], name='b_l1', scope=tik.scope_cbuf)
    dst_l1out = tik_instance.Tensor("float32", [15, 256, 16], name='dst_l1out', scope=tik.scope_cbuf_out)


    # 将数据搬至源操作数
    # tik_instance.data_move(a_l1, a_gm, 0, 1, 15360, 0, 0)
    tik_instance.data_move(b_l1, b_gm, 0, 1, 2048, 0, 0)
    # 进行matmul操作，mkn分别为32,64,160，dst_l1out的shape在m维度向上16对齐取值至32

    with tik_instance.for_range(0, 8*4) as outer_index:
        tik_instance.data_move(a_l1, a_gm[outer_index*8*256*16], 0, 1, 2048, 0, 0)
        tik_instance.matmul(dst_l1out, a_l1, b_l1, 256, 128, 240)
        tik_instance.fixpipe(dst_gm[outer_index*16*240*16], dst_l1out, 15, 512, 0, 0, extend_params={"relu": False})
            

    tik_instance.BuildCCE(kernel_name="matmul", inputs=[a_gm, b_gm], outputs=[dst_gm], output_files_path='.', config={"save_temp_cce_file": True})
    # tik_instance.BuildCCE(kernel_name="matmul", inputs=[a_gm, b_gm], outputs=[dst_gm], output_files_path='/home/zrji/old_method')




def matmul_tik_double_core():

    aicore_num = 2

    tik_instance = tik.Tik()

    a_gm = tik_instance.Tensor("float16", [8 * 1024*8*2 * 16], name='a_gm', scope=tik.scope_gm)
    b_gm = tik_instance.Tensor("float16", [8, 240, 16], name='b_gm', scope=tik.scope_gm)
    dst_gm = tik_instance.Tensor("float32", [16 * 1024*8*2 * 16], name='dst_gm', scope=tik.scope_gm)

    a_data_num_each_core = 8 * 1024*8 * 16
    c_data_num_each_core = 16 * 1024*8 * 16

    with tik_instance.for_range(
                0, aicore_num, block_num=aicore_num) as core_index:

        a_l1 = tik_instance.Tensor("float16", [8, 256, 16], name='a_l1', scope=tik.scope_cbuf)
        b_l1 = tik_instance.Tensor("float16", [8, 240, 16], name='b_l1', scope=tik.scope_cbuf)
        dst_l1out = tik_instance.Tensor("float32", [15, 256, 16], name='dst_l1out', scope=tik.scope_cbuf_out)

        tik_instance.data_move(b_l1, b_gm, 0, 1, 1920, 0, 0)

        with tik_instance.for_range(0, 8*4) as outer_index:
            tik_instance.data_move(a_l1, a_gm[core_index*a_data_num_each_core + outer_index*8*256*16], 0, 1, 2048, 0, 0)
            tik_instance.matmul(dst_l1out, a_l1, b_l1, 256, 128, 240)
            tik_instance.fixpipe(dst_gm[core_index*c_data_num_each_core+outer_index*16*240*16], dst_l1out, 15, 512, 0, 0, extend_params={"relu": False})

    tik_instance.BuildCCE(kernel_name="matmul", inputs=[a_gm, b_gm], outputs=[dst_gm], output_files_path='.', config={"save_temp_cce_file": True})
    # tik_instance.BuildCCE(kernel_name="matmul", inputs=[a_gm, b_gm], outputs=[dst_gm], output_files_path='/home/zrji/old_method')



def divide_tik():

    aicore_num = 2

    tik_instance = tik.Tik()

    src_gm = tik_instance.Tensor("float32", [12 * 1024 * 16], name='src_gm', scope=tik.scope_gm)
    dst_gm = tik_instance.Tensor("uint32", [12 * 1024 * 16], name='dst_gm', scope=tik.scope_gm)

    qp_gm = tik_instance.Tensor("float32", [192], name='qp_gm', scope=tik.scope_gm)
    qp_ub = tik_instance.Tensor("float32", [192], name='qp_ub', scope=tik.scope_ubuf)
    tik_instance.data_move(qp_ub, qp_gm, 0, 1, 30, 0, 0)

    qp_compute_ub = tik_instance.Tensor("float32", [64], name='qp_compute_ub', scope=tik.scope_ubuf)

    src_ub = tik_instance.Tensor("float32", [12, 256, 16], name='src_ub', scope=tik.scope_ubuf)

    src_ub16 = tik_instance.Tensor("float16", (256*16,), name='float16', scope=tik.scope_ubuf)
    one_ub = tik_instance.Tensor("float16", (256*16,), name="one_ub", scope=tik.scope_ubuf)
    tik_instance.vec_dup(128, one_ub, 0, 4, 8)
    
    dst_ub = tik_instance.Tensor("uint32", (256*6,), name='dst_ub', scope=tik.scope_ubuf)

    with tik_instance.for_range(0, 4) as outer_index:
        tik_instance.data_move(src_ub, src_gm[outer_index*12*256*16], 0, 1, 1536, 0, 0)
        
        #div query point norm
        with tik_instance.for_range(0, 12) as N_index:
            tik_instance.data_move(qp_compute_ub,     qp_ub[16*N_index], 0, 1, 2, 0, 0)
            tik_instance.data_move(qp_compute_ub[16], qp_ub[16*N_index], 0, 1, 2, 0, 0)
            tik_instance.data_move(qp_compute_ub[32], qp_ub[16*N_index], 0, 1, 2, 0, 0)
            tik_instance.data_move(qp_compute_ub[48], qp_ub[16*N_index], 0, 1, 2, 0, 0)
            tik_instance.vec_mul(64, src_ub[4096*N_index], src_ub[4096*N_index], qp_compute_ub, 64, 8, 8, 0)

        with tik_instance.for_range(0, 12) as N_index:
            tik_instance.vec_conv(64, "none", src_ub16, src_ub, 32, 4, 8)
            tik_instance.vec_cmpv_gt(dst_ub[128*N_index], src_ub16, one_ub, 32, 8, 8)

        tik_instance.data_move(dst_gm[outer_index*256*6], dst_ub, 0, 1, 24*4, 0, 0)
        
    tik_instance.BuildCCE(kernel_name="disdiv", inputs=[src_gm, qp_gm], outputs=[dst_gm], output_files_path='.', config={"save_temp_cce_file": True})

# tik_dprofile = tik.Dprofile()
# print(tik_dprofile.get_aicore_num())
# print(tik_dprofile.get_product_name())
matmul_tik_double_core()
divide_tik()




def divide_tik():

    aicore_num = 2

    tik_instance = tik.Tik()

    src_gm = tik_instance.Tensor("float32", [12 * 1024 * 16], name='src_gm', scope=tik.scope_gm)
    dst_gm = tik_instance.Tensor("float32", [12 * 1024 * 16], name='dst_gm', scope=tik.scope_gm)

    qp_gm = tik_instance.Tensor("float32", [192], name='qp_gm', scope=tik.scope_gm)
    qp_ub = tik_instance.Tensor("float32", [192], name='qp_ub', scope=tik.scope_ubuf)
    tik_instance.data_move(qp_ub, qp_gm, 0, 1, 30, 0, 0)

    qp_compute_ub = tik_instance.Tensor("float32", [64], name='qp_compute_ub', scope=tik.scope_ubuf)

    src_ub = tik_instance.Tensor("float32", [12, 256, 16], name='src_ub', scope=tik.scope_ubuf)

    src_ub16 = tik_instance.Tensor("float16", (256*16,), name='float16', scope=tik.scope_ubuf)
    one_ub = tik_instance.Tensor("float16", (256*16,), name="one_ub", scope=tik.scope_ubuf)
    tik_instance.vec_dup(128, one_ub, 32, 4, 8)


    # dst_ub = tik_instance.Tensor("uint8", [256], name='dst_ub', scope=tik.scope_ubuf)



    with tik_instance.for_range(0, 4) as outer_index:
        tik_instance.data_move(src_ub, src_gm[outer_index*12*256*16], 0, 1, 1536, 0, 0)
        
        #div query point norm
        with tik_instance.for_range(0, 12) as N_index:
            tik_instance.data_move(qp_compute_ub,     qp_ub[16*N_index], 0, 1, 2, 0, 0)
            tik_instance.data_move(qp_compute_ub[16], qp_ub[16*N_index], 0, 1, 2, 0, 0)
            tik_instance.data_move(qp_compute_ub[32], qp_ub[16*N_index], 0, 1, 2, 0, 0)
            tik_instance.data_move(qp_compute_ub[48], qp_ub[16*N_index], 0, 1, 2, 0, 0)
            tik_instance.vec_mul(64, src_ub[4096*N_index], src_ub[4096*N_index], qp_compute_ub, 64, 8, 8, 0)

        # tik_instance.vec_conv(64, "none", src_ub16, src_ub, 2, 4, 8)
        # tik_instance.vec_cmpv_gt(dst_ub, src_ub16, one_ub, 4, 8, 8)

        with tik_instance.for_range(0, 12) as N_index:
            tik_instance.vec_conv(64, "none", src_ub16, src_ub, 32, 4, 8)
            tik_instance.vec_cmpv_gt(dst_ub, src_ub16, one_ub, 32, 8, 8)


            # with tik_instance.for_range(0, 16) as M_index:
            #     tik_instance.vec_conv(64, "none", src_ub16, src_ub, 2, 4, 8)
            #     tik_instance.vec_cmpv_gt(dst_ub, src_ub16, one_ub, 4, 8, 8)


        tik_instance.data_move(dst_gm[outer_index*12*256*16], src_ub, 0, 1, 1536, 0, 0)
        






    tik_instance.BuildCCE(kernel_name="disdiv", inputs=[src_gm, qp_gm], outputs=[dst_gm], output_files_path='.', config={"save_temp_cce_file": True})