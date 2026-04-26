#pragma once

#include <vector>

std::vector<float> create_matrix(int n);

void fill_random(std::vector<float>& A);

void zero_matrix(std::vector<float>& A);

float max_abs_diff(const std::vector<float>& A,
                   const std::vector<float>& B);

void print_matrix(const std::vector<float>& A,
                  int n,
                  int max_rows = 6,
                  int max_cols = 6);
