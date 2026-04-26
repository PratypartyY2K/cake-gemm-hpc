#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                  \
    do                                                    \
    {                                                     \
        cudaError_t err = call;                           \
        if (err != cudaSuccess)                           \
        {                                                 \
            std::cerr << "CUDA error: "                   \
                      << cudaGetErrorString(err)          \
                      << " at line " << __LINE__ << "\n"; \
            std::exit(EXIT_FAILURE);                      \
        }                                                 \
    } while (0)

#define CHECK_CUBLAS(call)                       \
    do                                           \
    {                                            \
        cublasStatus_t status = call;            \
        if (status != CUBLAS_STATUS_SUCCESS)     \
        {                                        \
            std::cerr << "cuBLAS error at line " \
                      << __LINE__ << "\n";       \
            std::exit(EXIT_FAILURE);             \
        }                                        \
    } while (0)

std::vector<float> create_matrix(int n)
{
    return std::vector<float>(static_cast<size_t>(n) * n, 0.0f);
}

void fill_random(std::vector<float> &A)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &x : A)
    {
        x = dist(gen);
    }
}

void cpu_reference_gemm(const std::vector<float> &A,
                        const std::vector<float> &B,
                        std::vector<float> &C,
                        int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            float a = A[i * n + k];

            for (int j = 0; j < n; j++)
            {
                C[i * n + j] += a * B[k * n + j];
            }
        }
    }
}

float max_abs_diff(const std::vector<float> &A,
                   const std::vector<float> &B)
{
    float diff = 0.0f;

    for (size_t i = 0; i < A.size(); i++)
    {
        diff = std::max(diff, std::abs(A[i] - B[i]));
    }

    return diff;
}

double gflops(int n, float milliseconds)
{
    double ops = 2.0 * static_cast<double>(n) * n * n;
    double seconds = milliseconds / 1000.0;
    return ops / seconds / 1e9;
}

int main(int argc, char **argv)
{
    int n = 2048;

    if (argc >= 2)
    {
        n = std::atoi(argv[1]);
    }

    std::cout << "CAKE GEMM GPU cuBLAS Benchmark\n";
    std::cout << "N = " << n << "\n";

    std::vector<float> A = create_matrix(n);
    std::vector<float> B = create_matrix(n);
    std::vector<float> C = create_matrix(n);

    fill_random(A);
    fill_random(B);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm-up GEMM to mitigate startup overheads
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        d_B,
        n,
        d_A,
        n,
        &beta,
        d_C,
        n));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // cuBLAS assumes column-major layout.
    // For row-major C = A * B, compute equivalent column-major:
    // C^T = B^T * A^T
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        d_B,
        n,
        d_A,
        n,
        &beta,
        d_C,
        n));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nResults\n";
    std::cout << "gpu_time_ms,gpu_gflops\n";
    std::cout << milliseconds << "," << gflops(n, milliseconds) << "\n";

    if (n <= 1024)
    {
        std::vector<float> C_ref = create_matrix(n);
        cpu_reference_gemm(A, B, C_ref, n);
        float error = max_abs_diff(C, C_ref);
        std::cout << "max_error," << error << "\n";
    }
    else
    {
        std::cout << "max_error,skipped_for_large_N\n";
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}