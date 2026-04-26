#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>

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
    double ops = 2.0 * n * n * n;
    return ops / seconds / 1e9;
}

int main(int argc, char **argv)
{
    int n = 1024;
    int block_size = 64;

    if (argc >= 2)
        n = std::atoi(argv[1]);
    if (argc >= 3)
        block_size = std::atoi(argv[2]);

    std::cout << "CAKE GEMM CPU Benchmark\n";
    std::cout << "N = " << n << ", block size = " << block_size << "\n";

    auto A = create_matrix(n);
    auto B = create_matrix(n);
    auto C_naive = create_matrix(n);
    auto C_blocked = create_matrix(n);

    fill_random(A);
    fill_random(B);

    double t1 = now();
    naive_gemm(A, B, C_naive, n);
    double t2 = now();

    double naive_time = t2 - t1;

    t1 = now();
    blocked_gemm(A, B, C_blocked, n, block_size);
    t2 = now();

    double blocked_time = t2 - t1;

    float error = max_abs_diff(C_naive, C_blocked);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nResults\n";
    std::cout << "naive_time_sec,blocked_time_sec,naive_gflops,blocked_gflops,max_error\n";
    std::cout << naive_time << ","
              << blocked_time << ","
              << gflops(n, naive_time) << ","
              << gflops(n, blocked_time) << ","
              << error << "\n";

    return 0;
}