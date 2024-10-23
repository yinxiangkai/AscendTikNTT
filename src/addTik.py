from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np

def simple_add():
    # soc_version请设置为实际昇腾AI处理器的型号
    tbe_platform.set_current_compile_soc_info("Ascend310B1")
    tik_instance = tik.Tik(disable_debug=False)
    data_A = tik_instance.Tensor("float32", (64,2), name="data_A", scope=tik.scope_gm)
    data_B = tik_instance.Tensor("float32", (64,2), name="data_B", scope=tik.scope_gm)
    data_C = tik_instance.Tensor("float32", (64,2), name="data_C", scope=tik.scope_gm)
    data_A_ub = tik_instance.Tensor("float32", (128,), name="data_A_ub", scope=tik.scope_ubuf)
    data_B_ub = tik_instance.Tensor("float32", (128,), name="data_B_ub", scope=tik.scope_ubuf)
    data_C_ub = tik_instance.Tensor("float32", (128,), name="data_C_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(data_A_ub, data_A, 0, 1, 128*4//32, 0, 0)
    tik_instance.data_move(data_B_ub, data_B, 0, 1, 128*4//32, 0, 0)
    
    tik_instance.vec_add(64, data_C_ub[0], data_A_ub[0], data_B_ub[0], 2, 8, 8, 8)
    tik_instance.data_move(data_C, data_C_ub, 0, 1, 128*4//32, 0, 0)
    tik_instance.BuildCCE(kernel_name="simple_add",inputs=[data_A,data_B],outputs=[data_C])
    return tik_instance


if __name__ == "__main__":
    # 调用TIK算子实现函数
    tik_instance = simple_add()
    # 初始化数据，为128个float16类型的数字1的一维矩阵
    data = np.ones((64,2), dtype=np.float32)
    feed_dict = {"data_A": data, "data_B": data}
    # 启动功能调试
    data_C, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)
    # 打印输出数据
    print(data_C)