#!/bin/bash

# 清理 build 和 bin 目录
rm -rf build bin

# 初始化选项变量
OPTION1=""  # NVTX 默认关闭
OPTION2="-D DEBUG=OFF"  # 默认使用 Release 模式

# 解析传递的参数
for arg in "$@"
do
    case $arg in
        nvtx)
        OPTION1="-D NVTX=ON"
        shift # 从参数列表中移除 nvtx
        ;;
        debug)
        OPTION2="-D DEBUG=ON"
        shift # 从参数列表中移除 debug
        ;;
    esac
done

# 创建 build 目录并运行 cmake，传递处理过的选项
cmake -B build $OPTION1 $OPTION2
cmake --build build

# 修复 compile_commands.json 文件
JSON_FILE="./build/compile_commands.json"

# 根据调试和发布模式不同，调整 compile_commands.json 中的编译标志
if [ "$2" = "debug" ]; then
    sed -i 's/-arch=sm_90 -O3 -g -G/ --cuda-gpu-arch=sm_90 -O3 -g /g' "$JSON_FILE"
else
    sed -i 's/-arch=sm_90 -O3/ --cuda-gpu-arch=sm_90 /g' "$JSON_FILE"
fi
