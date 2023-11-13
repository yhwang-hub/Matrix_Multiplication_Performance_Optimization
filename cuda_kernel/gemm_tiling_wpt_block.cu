#include "../include/gemm.h"
#include <fstream>

static const int TILE_WIDTH = 64; // BLOCK_SIZE
static const int WPT = 4; // works per thread

__global__ void gemm_tiling_wpt_block_kernel(const int M, const int N, const int K, float* A, float* B, float* C)
{
    /* M×K * K×N = M×N */
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    Ads[threadIdx.y][threadIdx.x] = 0.;
    Bds[threadIdx.y][threadIdx.x] = 0.;

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    /* Init the acc registers */
    float acc[WPT][WPT];
    for (int wm = 0; wm < WPT; wm++)
    {
#pragma unroll
        for (int wn = 0; wn < WPT; wn++)
        {
            acc[wm][wn] = 0.0f;
        }
    }

    for (int ph = 0; ph < ceil(float(K) / float(TILE_WIDTH)); ph++)
    {
#pragma unroll
        for (int wm = 0; wm < WPT; wm++)
        {
#pragma unroll
            for (int wn = 0; wn < WPT; wn++)
            {
                /* Collaborative loading of A and B tiles into shared memory */
                if ((row + wm * (TILE_WIDTH / WPT)) < M && (ph * TILE_WIDTH + threadIdx.x + wn * (TILE_WIDTH / WPT)) < K)
                {
                    Ads[threadIdx.y + wm * (TILE_WIDTH / WPT)][threadIdx.x + wn * (TILE_WIDTH / WPT)] =
                        A[(row + wm * (TILE_WIDTH / WPT)) * K + ph * TILE_WIDTH + threadIdx.x + wn * (TILE_WIDTH / WPT)];
                }
                if ((ph * TILE_WIDTH + threadIdx.y + wm * (TILE_WIDTH / WPT)) < K && (col + wn * (TILE_WIDTH / WPT)) < N)
                {
                    Bds[threadIdx.y + wm * (TILE_WIDTH / WPT)][threadIdx.x + wn * (TILE_WIDTH / WPT)] =
                        B[(ph * TILE_WIDTH + threadIdx.y + wm * (TILE_WIDTH / WPT)) * N + col + wn * (TILE_WIDTH / WPT)];
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++)
        {
#pragma unroll
            for (int wm = 0; wm < WPT; wm++)
            {
#pragma unroll
                for (int wn = 0; wn < WPT; wn++)
                {
                    acc[wm][wn] += Ads[threadIdx.y + wm * (TILE_WIDTH / WPT)][i] * Bds[i][threadIdx.x + wn * (TILE_WIDTH / WPT)];
                    //debug bank confilct
                    /*int tid = threadIdx.y * (TILE_WIDTH/WPT) + threadIdx.x;
                    if(tid < 32){
                        int ads_bank_idx = ((threadIdx.y + wm*(TILE_WIDTH/WPT)) * TILE_WIDTH + i) ;//% 32;
                        int bds_bank_idx = (i * TILE_WIDTH + threadIdx.x + threadIdx.x + wn*(TILE_WIDTH/WPT)) % 32;
                        printf("tid=%d, ads_bank_idx=%d, bds_bank_idx=%d\n", tid, ads_bank_idx, bds_bank_idx);
                    }*/
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int wm = 0; wm < WPT; wm++)
    {
#pragma unroll
        for (int wn = 0; wn < WPT; wn++)
        {
            if ((row + wm * (TILE_WIDTH / WPT)) < M && (col + wn * (TILE_WIDTH / WPT)) < N)
            {
                C[(row + wm * (TILE_WIDTH / WPT)) * N + col + wn * (TILE_WIDTH / WPT)] = acc[wm][wn];
            }
        }   
    }
}


void gemm_gpu_tiling_wpt_block(const int M, const int K, const int N, float* A, float* B, float* C, int nIter)
{
    std::cout <<"My gemm gpu tiling wpt." << std::endl;
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
    dim3 dimBlock(TILE_WIDTH / WPT, TILE_WIDTH / WPT, 1);
    dim3 dimGrid(ceil(float(M) / float(dimBlock.x) / float(WPT)), ceil(float(N) / float(dimBlock.y) / float(WPT)), 1);
    std::cout<<dimBlock.x<<" "<<dimBlock.y<<" "<<dimGrid.x<<" "<<dimGrid.y<<std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float mseTotal = 0.;

    CUDA_CHECK(cudaEventRecord(start));
    for(int run = 0; run < nIter; ++run)
    {
        gemm_tiling_wpt_block_kernel<<<dimGrid, dimBlock, 0>>>(
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