cmake_minimum_required(VERSION 3.18)
project(cake_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenMP)

include_directories(include)

add_executable(cake_gemm
    src/main.cpp
    src/cpu/naive_gemm.cpp
    src/cpu/blocked_gemm.cpp
    src/utils/timer.cpp
    src/utils/matrix.cpp
)

target_link_libraries(cake_gemm OpenMP::OpenMP_CXX)

# GPU
add_executable(cake_gemm_gpu
    src/gpu/cublas_gemm.cu
)

target_link_libraries(cake_gemm_gpu cublas)
