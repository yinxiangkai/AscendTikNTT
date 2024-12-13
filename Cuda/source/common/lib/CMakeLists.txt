set(COMMON_SOURCE_FILES
    common.cu
    parameters.cu
)

add_library(common STATIC ${COMMON_SOURCE_FILES})

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)


set_target_properties(common PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)