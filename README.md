# Matrix Multiplication Performance Optimization
![Language](https://img.shields.io/badge/language-cuda-brightgreen)
![Language](https://img.shields.io/badge/CUDA-11.1-brightgreen)
![Language](https://img.shields.io/badge/ubuntu-16.04-brightorigin)

# Introduce
  This article will take single-precision matrix multiplication (Sgemm) as an example to discuss the optimization and acceleration of CUDA performance,
and use the basic knowledge of CUDA optimization to step by step optimize the performance of single-precision matrix multiplication to up to 70% of cublas.
The importance of matrix multiplication: In the field of high performance, the optimization of matrix multiplication (GEMM) is a very important topic.

  GEMM can be widely used in scientific computing fields such as aerospace and fluid mechanics, which was also the main application scenario of HPC before.
Later, deep learning developed in full swing, and due to the need for high computing power, it also became one of the main application scenarios of HPC.
A series of deep learning models have emerged in recent years. The most time-consuming things in the model, including convolution, fully connected layers, and attention, can all be converted into GEMM operations.
Therefore, the importance of GEMM optimization cannot be overemphasized.

This article explains[how to learn CUDA through matrix multiplication performance optimization](https://blog.csdn.net/qq_33287871/article/details/128280444?spm=1001.2014.3001.5502).

# Environment
The following environments have been tested：
- ubuntu16.04
- cuda11.1
- cudnn8.6.0
- cmake-3.24.0

# Build And Run
```
git clone git@github.com:yhwang-hub/Matrix_Multiplication_Performance_Optimization.git
cd Matrix_Multiplication_Performance_Optimization
mkdir build && cd build
cmake ..
make
./mat_multi
```

# Result
The comparison of the throughput results of the manually optimized kernel and CUDA matrix multiplication library cublas is shown in the following figure:
​![image](https://github.com/yhwang-hub/Matrix_Multiplication_Performance_Optimization/blob/main/images/Throughput_%20results%20_comparison.jpg)
[Note] The vertical axis is Gflop/s, and the horizontal axis is the matrix dimension.

The delay calculation method is the average of the sum of the times of calling CUDA kernel and Cublas 100 times respectively. The manually optimized kernel can reach up to 71% of the performance of Cublas.

# Reference
- https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda
- https://github.com/NervanaSystems/maxas/wiki/SGEMM
- https://zhuanlan.zhihu.com/p/410278370
- https://link.zhihu.com/?target=https%3A//github.com/Cjkkkk/CUDA_gemm
- https://link.zhihu.com/?target=https%3A//github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs
