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

/*
CAKE-style tiled GEMM idea:

C = A * B

We keep one C tile active and accumulate over K tiles:

for ii:
  for jj:
    C(ii,jj) = 0
    for kk:
      C(ii,jj) += A(ii,kk) * B(kk,jj)

This is not meant to beat full cuBLAS.
It is meant to expose communication/locality tradeoffs through controlled tiling.
*/
void tiled_cublas_gemm(cublasHandle_t handle,
                       const float *d_A,
                       const float *d_B,
                       float *d_C,
                       int n,
                       int tile)
{
    const float alpha = 1.0f;
    const float beta_first = 0.0f;
    const float beta_accum = 1.0f;

    for (int ii = 0; ii < n; ii += tile)
    {
        int m = std::min(tile, n - ii);

        for (int jj = 0; jj < n; jj += tile)
        {
            int cols = std::min(tile, n - jj);

            bool first_k = true;

            for (int kk = 0; kk < n; kk += tile)
            {
                int k = std::min(tile, n - kk);

                /*
                Row-major workaround using cuBLAS column-major interpretation.

                We want:
                    C[ii:ii+m, jj:jj+cols] +=
                    A[ii:ii+m, kk:kk+k] *
                    B[kk:kk+k, jj:jj+cols]

                cuBLAS sees row-major memory as transposed column-major.
                For full GEMM we used:
                    C^T = B^T * A^T

                For submatrices:
                    C_tile^T = B_tile^T * A_tile^T
                */

                const float *B_tile = d_B + static_cast<size_t>(kk) * n + jj;
                const float *A_tile = d_A + static_cast<size_t>(ii) * n + kk;
                float *C_tile = d_C + static_cast<size_t>(ii) * n + jj;

                CHECK_CUBLAS(cublasSgemm(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    cols, // rows of C^T
                    m,    // cols of C^T
                    k,    // inner dimension
                    &alpha,
                    B_tile,
                    n,
                    A_tile,
                    n,
                    first_k ? &beta_first : &beta_accum,
                    C_tile,
                    n));

                first_k = false;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int n = 2048;
    int tile = 512;

    if (argc >= 2)
        n = std::atoi(argv[1]);
    if (argc >= 3)
        tile = std::atoi(argv[2]);

    std::cout << "CAKE-style Tiled GPU GEMM Benchmark\n";
    std::cout << "N = " << n << ", tile = " << tile << "\n";

    std::vector<float> A = create_matrix(n);
    std::vector<float> B = create_matrix(n);
    std::vector<float> C = create_matrix(n);

    fill_random(A);
    fill_random(B);

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Warm-up
    tiled_cublas_gemm(handle, d_A, d_B, d_C, n, tile);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    tiled_cublas_gemm(handle, d_A, d_B, d_C, n, tile);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nResults\n";
    std::cout << "tiled_gpu_time_ms,tiled_gpu_gflops\n";
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
