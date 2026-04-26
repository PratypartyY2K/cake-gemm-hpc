#include "matrix.h"
#include <random>
#include <cmath>
#include <algorithm>

std::vector<float> create_matrix(int n)
{
    return std::vector<float>(n * n, 0.0f);
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

float max_abs_diff(const std::vector<float> &A, const std::vector<float> &B)
{
    float diff = 0.0f;

    for (size_t i = 0; i < A.size(); i++)
    {
        diff = std::max(diff, std::abs(A[i] - B[i]));
    }

    return diff;
}