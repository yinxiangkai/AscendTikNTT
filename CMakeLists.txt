cmake_minimum_required(VERSION 2.8.7)

project(NTT C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    include/head.cpp
)


add_executable(main ${SOURCES} )
target_include_directories(main PRIVATE /usr/local/Ascend/ascend-toolkit/latest/include)
target_link_libraries(main  /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libascendcl.so  
                              /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/libruntime.so   
                              /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libmsprofiler.so)

# baseline cpu
add_executable(baseline_cpu benchmark/baseline_cpu.cpp)
target_link_libraries(baseline_cpu gmp libgmpxx.so )
