# 示例程序说明

赛题提供了 slurm 作业脚本，选手修改代码后，直接执行 sbatch slurm-job.sh。脚本文
件在示例代码附件中，主要脚本中的路径是以示例代码压缩包解压到用户目录下。脚本包括
程序编译执行环境配置、编译及执行过程。脚本中的队列 ty_pd 可根据比赛时给定的队
列修改。

## 环境配置
测试算例使用DTK-22.10.1环境，命令如下: 
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/dtk/22.10.1
module list

## 程序编译    
进入示例程序目录，运行make gpu=1命令进行编译 
编译后的可执行程序为Csrsparse_rocsparse 
程序修改需要重新编译时，执行make clean gpu=1清理之前编译生成的文件再编译。

## 程序运行
示例程序运行需要提供稀疏矩阵 A 与稀疏矩阵 B 的信息。

矩阵 A 为随机生成，需要提供矩阵A随机生成标志0 ，矩阵A的行数、列数及稀疏度;
矩阵 B 为随机生成，默认矩阵 B 的行数、列数与矩阵 A 的列数相同，但需要提供稀疏度。

```shell
./Csrsparse_rocsparse 0  rows_A  cols_A  sparsity_A[0,1]  sparsity_B[0,1]  random_seed[1,-]

参数说明：
第一个参数为矩阵A生成方式，0表示随机生成；
矩阵A的行数；
矩阵A的列数；
矩阵A的稀疏度，范围0~1；
矩阵B的稀疏度，范围0~1；
生成矩阵A、B的随机数种子，范围>=1。
```

## 正确性验证
正确性验证主要时测试选手实现的计算函数返回的结果，与 common_function.hpp 中
给出的对照函数的结果对比，相对误差不大于 1e-9。

验证函数：
void verify(dtype* mat_kernel, dtype* mat_compare, size_t nnz)
对照函数：
void rocsparse_spgemm()

## 计算耗时显示
程序运行后会循环计算五次，给出每次的计算耗时和平均耗时，评分是主要查看平均耗时。


## 代码修改要求
1）参赛队在给定接口函数的基础上进行矩阵乘法性能优化，给定测试代码中，API接口函数call_device_spgemm()不可更改，示例代码中给出以伪随机数生成矩阵作为测试数据，验证算法的性能，具体接口见附录及示例demo程序。
2）参赛队需根据给出的demo程序对Csrsparse.hpp中的函数体进行具体实现与优化。需要注意：输入矩阵A、B以及输出矩阵C均采用稀疏矩阵CSR存储格式，输入矩阵A和B统一规定为非转置，输出矩阵C维度在函数运行前是未知的，因此在API接口函数call_device_spgemm()调用前不分配矩阵C的存储空间。需要在函数内分配矩阵C存储空间并返回矩阵C非0元个数，以及分配后的存储空间指针。相应的时间记录在总运行时间内，详细请参考《SpMM算子优化赛题示例程序附加说明》。
3）算法实现中如需分配临时存储空间，需要在函数内正确的处理空间分配与释放。参数选手需选取合理的算法，避免占用过多显存。
4）不允许采用稠密存储的方式处理，只能采用稀疏存储和算法，矩阵C非零元的估计不得超出实际值的3%。
5）参赛队不可以调用rocsparse及其他稀疏矩阵库函数接口，需完整实现所有的计算Kernel； API需要实现完整，要求在不同数据维度与稀疏度下均可计算正确。
6）禁止修改main.cpp和common_function.hpp等辅助与记时函数，其余源文件可以自定义，保证程序正确编译运行。
7）参数队需提供完整的源码、供赛事组进行评测。尽量保证只修改demo中的Csrsparse.hpp文件，不要添加过文件夹和文件；源码编译方式采用demo中给定的Makefile编译参数，不得修改。

需要实现的函数如下：
call_device_spgemm(transA, transB, alpha, m, n, k, 
		A_nnz, dptr_offset_A, dptr_colindex_A, dptr_value_A,
		B_nnz, dptr_offset_B, dptr_colindex_B, dptr_value_B,
		&C_nnz, dptr_offset_C, &pdptr_colindex_C, &pdptr_value_C);





ssh-keygen -f "/home/beck/.ssh/known_hosts" -R "[tycs.hpccube.com]:65101"

