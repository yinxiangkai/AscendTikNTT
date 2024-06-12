from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np

def get_platform_info(tbe_platform):
    blcok_num=tbe_platform.get_soc_spec("CORE_NUM")
    AICORE_TYPE = tbe_platform.get_soc_spec("AICORE_TYPE")
    UB_SIZE = tbe_platform.get_soc_spec("UB_SIZE")
    L2_SIZE = tbe_platform.get_soc_spec("L2_SIZE")
    L1_SIZE = tbe_platform.get_soc_spec("L1_SIZE")
    CUBE_SIZE = tbe_platform.get_soc_spec("CUBE_SIZE")
    L0A_SIZE = tbe_platform.get_soc_spec("L0A_SIZE")
    L0B_SIZE = tbe_platform.get_soc_spec("L0B_SIZE")
    L0C_SIZE = tbe_platform.get_soc_spec("L0C_SIZE")
    SMASK_SIZE = tbe_platform.get_soc_spec("SMASK_SIZE")

    print("CORE_NUM:",blcok_num)
    # 1
    print("AICORE_TYPE:",AICORE_TYPE)
    # aicore
    print("UB_SIZE:",UB_SIZE>>10,"KB")
    # 248
    print("L2_SIZE:",L2_SIZE>>20,"MB")
    # 4
    print("L1_SIZE:",L1_SIZE>>20,"MB")
    # 1
    print("CUBE_SIZE:",CUBE_SIZE)
    # 16 16 16
    print("L0A_SIZE:",L0A_SIZE>>10,"KB")
    # 64
    print("L0B_SIZE:",L0B_SIZE>>10,"KB")
    # 64
    print("L0C_SIZE:",L0C_SIZE>>10,"KB")
    # 128
    print("SMASK_SIZE:",SMASK_SIZE>>10,"KB")


    
        
def ntt_compute():
    tbe_platform.set_current_compile_soc_info("Ascend310B1")
    # get_platform_info(tbe_platform)
    blcok_num=tbe_platform.get_soc_spec("CORE_NUM")

    tik_instance = tik.Tik(disable_debug=False)
    range_size = tik_instance.InputScalar("int32",name="range_size")
    bit_size = tik_instance.InputScalar("int32",name="bit_size")
    group_size = tik_instance.InputScalar("int32",name="group_size")
    data_input_gm = tik_instance.Tensor("int8", (group_size,range_size), name="data_input_gm", scope=tik.scope_gm)
    power_table_gm = tik_instance.Tensor("int8", (group_size,range_size), name="power_table_gm", scope=tik.scope_gm)
    prime_gm = tik_instance.Tensor("int8", (group_size,), name="prime_gm", scope=tik.scope_gm)
    data_output_gm = tik_instance.Tensor("int8", (group_size,range_size), name="data_output_gm", scope=tik.scope_gm)
    

    with tik_instance.for_range(0, group_size) as group_id:  
        data_input_l1 = tik_instance.Tensor("int8", (range_size,), name="data_input_l1", scope=tik.scope_cbuf)
        power_table_l1 = tik_instance.Tensor("int8", (range_size,), name="power_table_l1", scope=tik.scope_cbuf)
        

            



    tik_instance.BuildCCE(kernel_name="",inputs=[data_input_gm,prime_gm,power_table_gm],outputs=[data_output_gm],flowtable=[range_size,bit_size,group_size],config={"save_temp_cce_file": True})
    return tik_instance


if __name__ == "__main__":
    

    # tik_instance = ntt_compute()
    # feed_dict = {"data_input_gm":data_input,"prime_gm":prime,"power_table_gm":power_tabel,"range_size":range_size,"bit_size":bit_size,"group_size":group_size}
    # 启动功能调试
   
    # data_output, = tik_instance.tikdb.start_debug(feed_dict=feed_dict, interactive=True)


    range_size = 1<<20
    root =1394649864822396625 
    prime_orign = 4179340454199820289
    bit_size = 7
    group_size = 9

    power_tabel_origin = np.ones(range_size, dtype=np.uint64)
    for i in range(1, range_size):
        power_tabel_origin[i] = (power_tabel_origin[i-1] * root) % prime_orign
    
    data_input_origin = np.ones((1024,1024), dtype=np.uint64)
    data_output_origin = np.ones((1024,1024), dtype=np.uint64)

    # 构建NTT矩阵
    NTTmatrix_origin = np.ones((1024,1024), dtype=np.uint64)
    for i in range(0, 1024):
        for j in range(0, 1024):
            NTTmatrix_origin[i][j] =power_tabel_origin[i*j*1024%range_size]
    
    data_input =np.zeros((9,1024,1024), dtype=np.int8)
    NTTmatrix =np.zeros((9,1024,1024), dtype=np.int8)
    prime =np.zeros(9, dtype=np.int8)

    #对数据进行拆分小端优先
    for i in range(1024):
        for j in range(1024):
            value = int(data_input_origin[i][j]) 
            for k in range(9):
                # 提取每个7位并转换为int8
                nibble = (value >> (k * 7)) & 0x7F
                data_input[k][i][j] = nibble
    
    for i in range(1024):
        for j in range(1024):
            value = int(data_input_origin[i][j]) 
            for k in range(9):
                # 提取每个7位并转换为int8
                nibble = (value >> (k * 7)) & 0x7F
                data_input[k][i][j] = nibble
    
     
    print(prime)
    
    



    for i in range(range_size):
        value = 0
        for j in range(group_size):
            # 将每个int8值组合回uint64
            value |= (data_output[j][i] & 0xF) << (j * 4)
        data_output_origin[i] = value
   
    # 打印输出数据
    print(data_output_origin)