#include "../include/gemm.h"
#include <fstream>

__global__ void gemm_naive_kernel(const int M, const int N, const int K, float* A, float* B, float* C)
{
    /* M×K * K×N = M×N */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= M || col >= N)
        return;

    float val = 0.0;
    for(int h = 0; h < K; ++h)
    {
        val += A[row * K + h] * B[h * N + col];
    }
    C[row * N + col] = val;
}


void gemm_gpu_naive(const int M, const int K, const int N, float* A, float* B, float* C, int nIter)
{
    std::cout <<"My gemm gpu naive." << std::endl;
    double flopsPerMatrixMul = 2.0 * M * N * K;

    /* 0. Malloc gpu for input & output */
    void* d_A = safeCudaMalloc(sizeof(float) * M * K);
    void* d_B = safeCudaMalloc(sizeof(float) * K * N);
    void* d_C = safeCudaMalloc(sizeof(float) * M * N);

    /* 1. Create cuda stream */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* 2. DMA(Direct Memory Access) the input to the GPU */
    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice, stream));

    /* 3. Launch kernel */
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil(float(M) / float(dimBlock.x)), ceil(float(N) / float(dimBlock.y)), 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float mseTotal = 0.;

    CUDA_CHECK(cudaEventRecord(start));
    for(int run = 0; run < nIter; ++run)
    {
        gemm_naive_kernel<<<dimGrid, dimBlock, 0>>>(
            M, K, N,
            static_cast<float*>(d_A),
            static_cast<float*>(d_B),
            static_cast<float*>(d_C)
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&mseTotal, start, stop));

    /* 4. Synchronize device and host */
    CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Cal kernel FLOPS */
    double msePerMatrixMul = mseTotal / nIter;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msePerMatrixMul / 1000.0f);
    printf(
        "My gemm Performance = %0.2f GFlops, Time = %0.3f mse, Size = %.0f Ops.\n",
        gigaFlops,
        msePerMatrixMul,
        flopsPerMatrixMul
    );

    /* Cal cublas FLOPS */
    cublasHandle_t blas_handle;
    checkCuBlasErrors(cublasCreate(&blas_handle));
    float alpha = 1.0;
    float beta = 0;
    CUDA_CHECK(cudaEventRecord(start));
    for(int run = 0; run < nIter; ++run)
    {
        checkCuBlasErrors (
            cublasSgemm(
                blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                M, N, K, &alpha,
                static_cast<float*>(d_A), M,
                static_cast<float*>(d_B), K,
                &beta,
                static_cast<float*>(d_C), K
            )
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&mseTotal, start, stop));
    std::vector<float> tmp(M * N);
    float* C1 = tmp.data();
    CUDA_CHECK(cudaMemcpyAsync(C1, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost, stream));

    msePerMatrixMul = mseTotal / nIter;
    gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msePerMatrixMul / 1000.0f);
    printf(
        "My gemm Performance = %0.2f GFlops, Time = %0.3f mse, Size = %.0f Ops.\n",
        gigaFlops,
        msePerMatrixMul,
        flopsPerMatrixMul
    );

    double eps = 1.e-6;
    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        // C1 is transpose
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(C[i] - C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if(rel_err > eps)
        {
            printf(
                "Error! Matrix[%05d] = %.8f, ref = %.8f error term is > %E\n",
                i, C[i], C1[col * M + row], eps
            );
            correct = false;
            break;
        }
    }
    printf("Correct = %d\n", correct);

    /* 5. Destroy & Clean */
    cublasDestroy(blas_handle);
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}