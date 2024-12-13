set(LIB_SOURCE_FILES
  tensor_0.cu
  tensor_1.cu
  tensor_2.cu
  tensor_3.cu
  tensor_4.cu
  tensor_5.cu
  tensor_6.cu
  tensor_7.cu
  tensor_8.cu
)

add_library(tensor_ntt STATIC ${LIB_SOURCE_FILES})

target_link_libraries(tensor_ntt PUBLIC common)

set_target_properties(tensor_ntt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(tensor_ntt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
