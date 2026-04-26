#include <vector>

void naive_gemm(const std::vector<float> &A,
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