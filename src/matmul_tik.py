from tbe import tik
tik_instance = tik.Tik()
# 定义tensor
a_gm = tik_instance.Tensor("int8", [2, 32, 32], name='a_gm', scope=tik.scope_gm)
b_gm = tik_instance.Tensor("int8", [2, 160, 32], name='b_gm', scope=tik.scope_gm)
# 由于matmul矩阵计算m=30，fixpipe会将dst_l1out中无效数据删除，因此dst_gm的shape在m方向上设置为30即可
dst_gm = tik_instance.Tensor("int32", [10, 30, 16], name='dst_gm', scope=tik.scope_gm)
a_l1 = tik_instance.Tensor("int8", [2, 32, 32], name='a_l1', scope=tik.scope_cbuf)
b_l1 = tik_instance.Tensor("int8", [2, 160, 32], name='b_l1', scope=tik.scope_cbuf)
dst_l1out = tik_instance.Tensor("int32", [10, 32, 16], name='dst_l1out', scope=tik.scope_cbuf_out)
# 将数据搬至源操作数
tik_instance.data_move(a_l1, a_gm, 0, 1, 64, 0, 0)
tik_instance.data_move(b_l1, b_gm, 0, 1, 320, 0, 0)
# 进行matmul操作，mkn分别为30,64,160，dst_l1out的shape在m维度向上16对齐取值至32
tik_instance.matmul(dst_l1out, a_l1, b_l1, 30, 64, 160)
# 将数据搬移至dst_gm，其中burst_len = 30*16*dst_l1out_dtype_size//32 = 60
tik_instance.fixpipe(dst_gm, dst_l1out, 10, 60, 0, 0, extend_params={"relu": True})
tik_instance.BuildCCE(kernel_name="matmul", inputs=[a_gm, b_gm], outputs=[dst_gm])