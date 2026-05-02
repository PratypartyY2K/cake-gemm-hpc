#include <iostream>
#include <cuda_runtime.h>

#define CHECK(call) \
    if (call != cudaSuccess) { \
        std::cerr << "CUDA error\n"; exit(1); \
    }

int main() {
    size_t N = 1LL << 28; // ~256M elements (~1GB)
    size_t bytes = N * sizeof(float);

    float *d_a, *d_b;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice);

    cudaEventRecord(start);

    for (int i = 0; i < 10; i++) {
        cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double time_sec = ms / 1000.0;
    double total_bytes = bytes * 10;

    double bandwidth = total_bytes / time_sec / 1e9; // GB/s

    std::cout << "Bandwidth (GB/s): " << bandwidth << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
}
