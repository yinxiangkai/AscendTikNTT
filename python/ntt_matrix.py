from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np

def ntt_compute():
    tbe_platform.set_current_compile_soc_info("Ascend310B1")
    tik_instance = tik.Tik(disable_debug=False)
    size = tik_instance.InputScalar("int32",name="size")
    data_input_gm = tik_instance.Tensor("int8", (16,size), name="data_input_gm", scope=tik.scope_gm)
    power_table_gm = tik_instance.Tensor("int8", (16,size), name="power_table_gm", scope=tik.scope_gm)
    prime_gm = tik_instance.Tensor("int8", (16,), name="prime_gm", scope=tik.scope_gm)
    data_output_gm = tik_instance.Tensor("int8", (16,size), name="data_output_gm", scope=tik.scope_gm)
    



    tik_instance.BuildCCE(kernel_name="ntt_matrix",inputs=[data_input_gm,prime_gm,power_table_gm],outputs=[data_output_gm],flowtable=[size],config={"save_temp_cce_file": True})
    return tik_instance


if __name__ == "__main__":
    
    size = 1<<20
    root =3 
    prime_orign = 4179340454199820289

    data_input_origin = np.ones(size, dtype=np.uint64)
    data_output_origin = np.ones(size, dtype=np.uint64)
    power_tabel_origin = np.ones(size, dtype=np.uint64)
    for i in range(1, size):
        power_tabel_origin[i] = (power_tabel_origin[i-1] * root) % prime_orign
    

    data_input = np.zeros((16,size), dtype=np.int8)
    data_output = np.zeros((16,size), dtype=np.int8)
    power_tabel = np.zeros((16,size), dtype=np.int8)
    prime = np.zeros(16, dtype=np.int8)

    for i in range(size):
        value = int(data_input_origin[i]) 
        for j in range(16):
            # 提取每个4位 (半字节) 并转换为int8
            nibble = (value >> (j * 4)) & 0xF
            data_input[j][i] = nibble

    for i in range(size):
        value = int(power_tabel_origin[i]) 
        for j in range(16):
            # 提取每个4位 (半字节) 并转换为int8
            nibble = (value >> (j * 4)) & 0xF
            power_tabel[j][i] = nibble

    for j in range(16):
            # 提取每个4位 (半字节) 并转换为int8
            nibble = (prime_orign >> (j * 4)) & 0xF
            prime[j] = nibble

    feed_dict = {"data_input_gm":data_input,"prime_gm":prime,"power_table_gm":power_tabel,"size":size}
    # 启动功能调试
    tik_instance = ntt_compute()
    data_output, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)


    for i in range(size):
        value = 0
        for j in range(16):
            # 将每个int8值组合回uint64
            value |= (data_output[j][i] & 0xF) << (j * 4)
        data_output_origin[i] = value
   
    # 打印输出数据
    print(data_output_origin)