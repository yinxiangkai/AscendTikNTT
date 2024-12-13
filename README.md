
# NTT算子实现

## 总览

本项目主要是实现NTT在矩阵计算单元上的实现，为了验证算法的适用性，分别在Nvidia GPU，Ascend Atlas上进行了测试。
主要是完成了如下功能：
* 提供了多种七种NTT实现。
* 批量NTT算法实现。
* 基于GPU的大数模约简方案。
* 基于Ascend 的大整数拆分方案。
* 基于Ascend 的NTT方案。

## 文档结构说明
代码目录结构如下主要分为两部分，分别是cuda 的实现和Ascend的实现。

AscendTikNTT
├─ Ascend
│  ├─ CMakeLists.txt
│  ├─ benchmark
│  ├─ build
│  ├─ include
│  ├─ kernel_meta
│  ├─ matrix.txt
│  ├─ out
│  ├─ python
│  ├─ run.sh
│  ├─ src
│  └─ test
├─ Cuda
│  ├─ CMakeLists.txt
│  ├─ benchmark
│  ├─ run.sh
│  └─ source
│     ├─ CMakeLists.txt
│     ├─ common
│     ├─ merge_ntt 
│     └─ tensor_ntt  
├─ License
└─ README.md

## 环境部署
|环境|硬件|软件|
| --------| ------------------| -----------|
|Ascend|Atlas 200I DK A2|CANN23.0|
|Nvidia|H100|Cuda 12.6|

Ascend的环境配置参考：https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC2/qs/qs_0002.html

## 使用
### 1 编译
每个目录下面均有run.sh 文件,运行下列命令即可进行编译。
```
bash run.sh
```
### 2 引用
cuda 代码，将source 目录添加到项目中然后就可以使用大整数模乘方案，NTT算子。
Ascend 代码，将kernel_meta目录中的算子添加到自己的项目中，在项目加载核函数即可。
