# CAKE-Inspired Communication-Avoiding GEMM with GPU Acceleration

## Overview

This project explores communication-avoiding matrix multiplication using a CAKE-inspired tiled GEMM approach with GPU acceleration.

The objective is to study how tiling and scheduling strategies affect performance, particularly in terms of:

- data locality
- GPU utilization
- kernel launch overhead
- arithmetic intensity

We compare a CAKE-style tiled approach against a highly optimized cuBLAS baseline.

---

## Motivation

Matrix multiplication is a fundamental building block in:

- scientific computing
- numerical linear algebra
- deep learning
- high-performance computing (HPC)

Modern GPUs provide massive compute capability, but performance is often limited by:

- memory bandwidth
- data movement
- kernel scheduling overhead

This project investigates how communication-avoiding ideas (CAKE-style tiling) influence performance on GPU architectures.

---

## System Configuration

Experiments were conducted on:

- GPU: NVIDIA RTX A4500
- CUDA: 12.4
- Compiler: GCC 11.3
- Build system: CMake
- Library: cuBLAS

---

## Implemented Methods

### 1. CPU Naive GEMM

Basic triple-loop implementation.

Used only for correctness and baseline comparison.

---

### 2. CPU Blocked GEMM

Cache-aware implementation using:

- blocking (tiling)
- OpenMP parallelism

Improves memory locality and CPU utilization.

---

### 3. Full GPU cuBLAS GEMM

Uses cublasSgemm().

This is the performance baseline, representing highly optimized vendor implementation.

---

### 4. CAKE-Style Tiled GPU GEMM

A tiled GEMM approach inspired by communication-avoiding principles.

Structure:

for each C tile:
keep C tile active
for each K tile:
C_tile += A_tile × B_tile

Key idea:

- reuse data within tiles
- reduce unnecessary data movement
- expose locality vs overhead tradeoff

Each tile multiplication is performed using cuBLAS.

---

## Repository Structure

cake-gemm/
├── CMakeLists.txt
├── README.md
├── include/
│ ├── matrix.h
│ └── timer.h
├── src/
│ ├── main.cpp
│ ├── cpu/
│ │ ├── naive_gemm.cpp
│ │ └── blocked_gemm.cpp
│ ├── gpu/
│ │ ├── cublas_gemm.cu
│ │ └── tiled_gemm.cu
│ └── utils/
│ ├── matrix.cpp
│ └── timer.cpp
├── scripts/
│ └── plot_results.py
└── results/
├── summary.csv
└── plots/

---

## Build Instructions

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:/home/software/gcc/gcc-11.3.0/bin:$PATH
export LD_LIBRARY_PATH=/home/software/gcc/gcc-11.3.0/lib64:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

rm -rf build
mkdir build
cd build

cmake ..
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
-DCMAKE_CUDA_ARCHITECTURES=86

make -j

---

## Run Instructions

CPU:

./cake_gemm 1024 64

GPU (cuBLAS baseline):

./cake_gemm_gpu 4096
./cake_gemm_gpu 8192
./cake_gemm_gpu 16384

CAKE-style tiled GPU:

./cake_gemm_tiled_gpu 4096 512
./cake_gemm_tiled_gpu 4096 1024
./cake_gemm_tiled_gpu 4096 2048

./cake_gemm_tiled_gpu 8192 1024
./cake_gemm_tiled_gpu 8192 2048
./cake_gemm_tiled_gpu 8192 4096

---

## Results Summary

N = 4096
Full cuBLAS: ~15016 GFLOP/s
Tiled 1024: ~13103 GFLOP/s
Tiled 2048: ~13344 GFLOP/s

N = 8192
Full cuBLAS: ~15427 GFLOP/s
Tiled 1024: ~11364 GFLOP/s
Tiled 2048: ~12074 GFLOP/s
Tiled 4096: ~12562 GFLOP/s

N = 16384
Full cuBLAS: ~15106 GFLOP/s
Tiled 1024: ~11109 GFLOP/s
Tiled 2048: ~11764 GFLOP/s
Tiled 4096: ~12229 GFLOP/s

---

## Key Observations

- Full cuBLAS achieves peak performance (~15 TFLOP/s).
- Tiled GEMM improves as tile size increases.
- Small tiles suffer from overhead (many cuBLAS calls).
- Large tiles improve arithmetic intensity and GPU utilization.
- Tiled approach approaches but does not surpass full cuBLAS.

---

## Conclusion

This project demonstrates the core tradeoff in communication-avoiding algorithms:

- smaller tiles → more flexibility but higher overhead
- larger tiles → better performance but reduced flexibility

The CAKE-style tiled implementation effectively illustrates how scheduling and locality influence performance on GPUs.

---

## Future Work

- CUDA streams (overlap execution)
- double buffering
- asynchronous tile execution
- MPI-based distributed GEMM
- roofline performance analysis
