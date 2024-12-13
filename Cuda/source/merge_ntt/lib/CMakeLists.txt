set(LIB_SOURCE_FILES
  merge_ntt_cpu.cu
  merge_ntt.cu
)

add_library(merge_ntt STATIC ${LIB_SOURCE_FILES})

target_link_libraries(merge_ntt PUBLIC common)

set_target_properties(merge_ntt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(merge_ntt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
