/**
* @Author : yuehong.Wang 
* @DAte   : 2023-11-08
* @Brief  : Matrix multiply performance optimization
*/

#include "./include/gemm.h"
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

/* for creating rand numbers */
#include <cstdlib>
#include <ctime>

static const int M = 2048;
static const int K = 2048;
static const int N = 2048;

void mat_multi_cpu(const int M, const int K, const int N, float* A, float* B, float* C)
{
    /* M×N * K×N = M×N */
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float val = 0.0;
            for (int h = 0; h < K; h++)
                val += A[i * K + h] * B[j + h * N];
            C[i * N + j] = val;
        }
    }
}

void print_matrix(float* C, int M, int N)
{
    std::cout << "[" <<std::endl;
    for (int i = 0; i < M; i++)
    {
        std::cout << " ";
        for (int j = 0; j < N; j++)
        {
            float val = C[i * N + j];
            std::cout << val << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << " ]" <<std::endl;
}

int main(int argc, char** argv)
{
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    std::vector<float> D(M * N);

    /* Init vector A & vector B by rand(), each value's range is 0~rBound */
    srand(static_cast<unsigned>(time(0)));
    float rBound = 5;
    for (int i = 0; i < M * K; i++)
    {
        float rr = static_cast<float> (rand() / (static_cast<float>(RAND_MAX / rBound)));
        A[i] = rr;
    }
    for (int i = 0; i < K * N; i++)
    {
        float rr = static_cast<float> (rand() / (static_cast<float>(RAND_MAX / rBound)));
        B[i] = rr;
    }

    float* a = A.data();
    float* b = B.data();
    float* c = C.data();
    float* d = D.data();

    std::cout << "Matrix size: " << M << " x " << N << std::endl;
    std::cout << "Runing matrix multiply on gpu..." << std::endl;
    gemm_gpu_naive(M, K, N, a, b, d, 100);
    gemm_gpu_tiling(M, K, N, a, b, d, 100);
    gemm_gpu_tiling_wpt(M, K, N, a, b, d, 100);
    gemm_gpu_tiling_wpt_block(M, K, N, a, b, d, 100);
    gemm_gpu_tiling_wpt_block_rect(M, K, N, a, b, d, 100);

    /* Debug results */
    /* std::cout << "Runing matrix multiply on cpu..." << std::endl;
    mat_multi_cpu(M, K, N, a, b, c);
    std::cout<<"--------A--------"<<std::endl;
    print_matrix(a, M, K);
    std::cout<<"--------B--------"<<std::endl;
    print_matrix(b, K, N);
    std::cout<<"--------C--------"<<std::endl;
    print_matrix(d, M, N);
    std::cout<<"--------Ref--------"<<std::endl;
    print_matrix(c, M, N); */

    return 0;
}