#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

#include "matrix.h"
#include "timer.h"

void naive_gemm(const std::vector<float> &A,
                const std::vector<float> &B,
                std::vector<float> &C,
                int n);

void blocked_gemm(const std::vector<float> &A,
                  const std::vector<float> &B,
                  std::vector<float> &C,
                  int n,
                  int block_size);

double gflops(int n, double seconds)
{
    double ops = 2.0 * static_cast<double>(n) * n * n;
    return ops / seconds / 1e9;
}

int main(int argc, char **argv)
{
    int n = 1024;
    int block_size = 64;
    int run_naive = 1;

    if (argc >= 2)
        n = std::atoi(argv[1]);
    if (argc >= 3)
        block_size = std::atoi(argv[2]);
    if (argc >= 4)
        run_naive = std::atoi(argv[3]);

    std::cout << "CAKE-style CPU OpenMP GEMM Benchmark\n";
    std::cout << "N = " << n
              << ", block size = " << block_size
              << ", OpenMP threads = " << omp_get_max_threads()
              << "\n";

    auto A = create_matrix(n);
    auto B = create_matrix(n);
    auto C_blocked = create_matrix(n);

    fill_random(A);
    fill_random(B);

    double naive_time = -1.0;
    double naive_perf = -1.0;
    float error = -1.0f;

    std::vector<float> C_naive;

    if (run_naive)
    {
        C_naive = create_matrix(n);

        double t1 = now();
        naive_gemm(A, B, C_naive, n);
        double t2 = now();

        naive_time = t2 - t1;
        naive_perf = gflops(n, naive_time);
    }

    double t1 = now();
    blocked_gemm(A, B, C_blocked, n, block_size);
    double t2 = now();

    double blocked_time = t2 - t1;
    double blocked_perf = gflops(n, blocked_time);

    if (run_naive)
    {
        error = max_abs_diff(C_naive, C_blocked);
    }

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nResults\n";
    std::cout << "N,block_size,threads,naive_time_sec,blocked_time_sec,naive_gflops,blocked_gflops,max_error\n";
    std::cout << n << ","
              << block_size << ","
              << omp_get_max_threads() << ","
              << naive_time << ","
              << blocked_time << ","
              << naive_perf << ","
              << blocked_perf << ","
              << error << "\n";

    return 0;
}