# CAKE-Inspired Communication-Avoiding GEMM with GPU Acceleration

## Overview
This project explores communication-avoiding matrix multiplication
using CAKE-style blocking and GPU acceleration via CUDA/cuBLAS.

## Features
- CPU naive and blocked GEMM
- CAKE-inspired tiled GEMM
- GPU acceleration using cuBLAS
- Performance benchmarking and analysis

## Build
```bash
mkdir build && cd build
cmake ..
make -j
