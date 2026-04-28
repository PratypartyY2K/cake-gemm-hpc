import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path("/Users/pratyushkumar/Desktop/Pratyush/PennState/Spring 2026/Concurrent/Project/cake-gemm-hpc/results/cpu_summary.csv")
PLOTS_DIR = Path("/Users/pratyushkumar/Desktop/Pratyush/PennState/Spring 2026/Concurrent/Project/cake-gemm-hpc/results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

# Plot 1: OpenMP scaling, fixed N=1024, block=64
scaling = df[(df["N"] == 1024) & (df["block_size"] == 64)].copy()
scaling = scaling.sort_values("threads")

plt.figure()
plt.plot(scaling["threads"], scaling["gflops"], marker="o")
plt.xlabel("OpenMP Threads")
plt.ylabel("GFLOP/s")
plt.title("CPU OpenMP Scaling, N=1024, block=64")
plt.grid(True)
plt.savefig(PLOTS_DIR / "cpu_openmp_scaling.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: CPU block-size sensitivity, fixed N=2048, threads=8
tiles = df[(df["N"] == 2048) & (df["threads"] == 8)].copy()
tiles = tiles.sort_values("block_size")

plt.figure()
plt.plot(tiles["block_size"], tiles["gflops"], marker="o")
plt.xlabel("Block Size")
plt.ylabel("GFLOP/s")
plt.title("CPU Block Size Sensitivity, N=2048, threads=8")
plt.grid(True)
plt.savefig(PLOTS_DIR / "cpu_block_size_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close()

print("CPU plots saved to results/plots/")