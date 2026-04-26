import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path("results/summary.csv")
PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

# ----------------------------
# Plot 1: GFLOP/s vs tile size
# ----------------------------
tiled = df[df["method"] == "tiled_cublas"].copy()
tiled["tile"] = tiled["tile"].astype(int)

for n in sorted(tiled["N"].unique()):
    subset = tiled[tiled["N"] == n]

    plt.figure()
    plt.plot(subset["tile"], subset["gflops"], marker="o")
    plt.xlabel("Tile size")
    plt.ylabel("GFLOP/s")
    plt.title(f"CAKE-style tiled GEMM performance, N={n}")
    plt.grid(True)
    plt.savefig(PLOTS_DIR / f"gflops_vs_tile_N{n}.png", dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------------
# Plot 2: Full cuBLAS vs best tiled
# -----------------------------------
full = df[df["method"] == "full_cublas"].copy()
best_tiled = (
    tiled.sort_values("gflops", ascending=False)
    .groupby("N")
    .first()
    .reset_index()
)

comparison = pd.DataFrame({
    "N": full["N"].values,
    "full_cublas_gflops": full["gflops"].values,
    "best_tiled_gflops": best_tiled["gflops"].values,
    "best_tile": best_tiled["tile"].values,
})

plt.figure()
plt.plot(comparison["N"], comparison["full_cublas_gflops"], marker="o", label="Full cuBLAS")
plt.plot(comparison["N"], comparison["best_tiled_gflops"], marker="o", label="Best tiled cuBLAS")
plt.xlabel("Matrix size N")
plt.ylabel("GFLOP/s")
plt.title("Full cuBLAS vs best CAKE-style tiled GEMM")
plt.grid(True)
plt.legend()
plt.savefig(PLOTS_DIR / "full_vs_best_tiled_gflops.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------
# Plot 3: Runtime vs matrix size
# -------------------------------
plt.figure()
plt.plot(full["N"], full["time_ms"], marker="o", label="Full cuBLAS")
plt.plot(best_tiled["N"], best_tiled["time_ms"], marker="o", label="Best tiled cuBLAS")
plt.xlabel("Matrix size N")
plt.ylabel("Runtime (ms)")
plt.title("Runtime comparison")
plt.grid(True)
plt.legend()
plt.savefig(PLOTS_DIR / "runtime_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plots saved to results/plots/")
print(comparison)
