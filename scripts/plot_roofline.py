import numpy as np
import matplotlib.pyplot as plt

peak_flops = 15000  # GFLOP/s
bandwidth = 285     # GB/s

intensity = np.logspace(-1, 4, 100)

roofline = np.minimum(peak_flops, bandwidth * intensity)

plt.figure()
plt.loglog(intensity, roofline, label="Roofline")

Ns = [1024, 4096, 8192, 16384]
gflops = [12000, 15000, 15400, 15100]

intensities = [n / 6 for n in Ns]

plt.scatter(intensities, gflops, label="GEMM", color="red")

plt.xlabel("Operational Intensity (FLOPs/byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Roofline Model")
plt.legend()
plt.grid(True)

plt.savefig("results/plots/roofline.png", dpi=300)
plt.show()
