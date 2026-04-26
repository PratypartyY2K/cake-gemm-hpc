cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 86)

project(cake_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(include)

add_executable(cake_gemm
    src/main.cpp
    src/cpu/naive_gemm.cpp
    src/cpu/blocked_gemm.cpp
    src/utils/timer.cpp
    src/utils/matrix.cpp
)

target_link_libraries(cake_gemm PRIVATE OpenMP::OpenMP_CXX)

add_executable(cake_gemm_gpu
    src/gpu/cublas_gemm.cu
)

target_link_libraries(cake_gemm_gpu PRIVATE CUDA::cublas CUDA::cudart)