add_executable(merge_time merge_time.cu)
target_link_libraries(merge_time PRIVATE merge_ntt)
set_target_properties(merge_time PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(tensor_0_time tensor_0_time.cu)
target_link_libraries(tensor_0_time PRIVATE tensor_ntt)

add_executable(tensor_1_time tensor_1_time.cu)
target_link_libraries(tensor_1_time PRIVATE tensor_ntt)

add_executable(tensor_2_time tensor_2_time.cu)
target_link_libraries(tensor_2_time PRIVATE tensor_ntt)

add_executable(tensor_3_time tensor_3_time.cu)
target_link_libraries(tensor_3_time PRIVATE tensor_ntt)

add_executable(tensor_4_time tensor_4_time.cu)
target_link_libraries(tensor_4_time PRIVATE tensor_ntt)

add_executable(tensor_5_time tensor_5_time.cu)
target_link_libraries(tensor_5_time PRIVATE tensor_ntt)

add_executable(tensor_6_time tensor_6_time.cu)
target_link_libraries(tensor_6_time PRIVATE tensor_ntt)

add_executable(tensor_7_time tensor_7_time.cu)
target_link_libraries(tensor_7_time PRIVATE tensor_ntt)

add_executable(tensor_8_time tensor_8_time.cu)
target_link_libraries(tensor_8_time PRIVATE tensor_ntt)

add_executable(combine combine.cu)
target_link_libraries(combine PRIVATE tensor_ntt merge_ntt )