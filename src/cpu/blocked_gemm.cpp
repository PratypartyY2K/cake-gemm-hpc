#include <vector>
#include <algorithm>
#include <omp.h>

void blocked_gemm(const std::vector<float> &A,
                  const std::vector<float> &B,
                  std::vector<float> &C,
                  int n,
                  int block_size)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += block_size)
    {
        for (int jj = 0; jj < n; jj += block_size)
        {
            for (int kk = 0; kk < n; kk += block_size)
            {

                int i_max = std::min(ii + block_size, n);
                int j_max = std::min(jj + block_size, n);
                int k_max = std::min(kk + block_size, n);

                for (int i = ii; i < i_max; i++)
                {
                    for (int k = kk; k < k_max; k++)
                    {
                        float a = A[i * n + k];

#pragma omp simd
                        for (int j = jj; j < j_max; j++)
                        {
                            C[i * n + j] += a * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}