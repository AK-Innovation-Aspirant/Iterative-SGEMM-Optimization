# Iterative SGEMM Optimization

This project documents the iterative optimization of FP32 SGEMM (Single-Precision General Matrix Multiplication) for large square matrices (4092x4092) using CUDA. It follows the progression based on the highly cited Siboehm Blog (link below) and heavily uses code from it adapted to my GPU setup. The goal was to learn the fundamentals of low-level CUDA programming and extend it to a real-world problem. This was a personal side-project done during my internship **Intel Labs** with the GPU used being a **QUADRO RTX 5000**.

> ⚠️ **Note:** The actual implementation developed on Intel workstations is considered **Intel IP** and cannot be shared. However, this project replicates and builds upon open-source resources for educational and experimental purposes.

---

## 📚 Learning Resources

The optimization techniques used were informed by the following resources:

- 🎓 [FreeCodeCamp CUDA Crash Course](https://www.youtube.com/watch?v=86FAWCzIe_4)
- 💻 [FreeCodeCamp CUDA Crash Course Github](https://github.com/Infatoshi/cuda-course)  
- 🧪 [OLCF CUDA Training Series](https://www.youtube.com/playlist?list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj)  
- ✍️ [CUDA Matrix Multiplication Optimization Blog by Simon Böehm](https://siboehm.com/articles/22/CUDA-MMM)  
- 💻 [Blog Associated GitHub Repository](https://github.com/siboehm/SGEMM_CUDA/tree/master)

---

## 📊 Performance Table

The GPU used for all benchmarks was an **NVIDIA Quadro RTX 5000**, which has a **theoretical FP32 compute throughput of ~11.2 TFLOPS**.  
For a matrix size of **4092×4092**, the total operation count for SGEMM is:

Operation count = 2 × (4092³) ≈ 1.37 × 10¹¹ floating-point operations (FLOPs) / 137 GFLOPS

At the theoretical maximum, this would correspond to a runtime of roughly **12.2 ms**—a useful lower bound for comparing against measured results.

Below is a summary of kernel variants, their implementation focus, and execution time:

| Kernel Number   | Name                     | Timing (M,K,N=4092)                        | Approx. GFLOP/s |
|-----------------|--------------------------|--------------------------------------------|-----------------|
| 1               | Naive Implementation     | 615.086 ms (1d)                            | ~223            |
| 2               | GMEM Coalescing          | 134.786 ms (2d layout)                     | ~1018           |
| 3               | Shared Memory Access     | 103.786 ms (1d)                            | ~1322           |
| 4               | 1D Blocktiling           | 48.255 ms (1d)                             | ~2842           |
| 5               | 2D Blocktiling           | 21.8304 ms (2d layout) / 23.4354 ms (1d)   | ~6285 / ~5852   |
| 6               | Vectorized Access        | 22.0411 ms (1d)                            | ~6220           |
| 7               | Bank Conflicts Extra Col | 26.7314 ms (1d)                            | ~5128           |
| 8               | Bank Conflicts Swizzling | 18.6528 ms (1d)                            | ~7345           |
| 9               | Autotuning               | 18.1215 ms (no swizzle, see notes for Kernel 9) | ~7557       |
| 10              | Warptiling               | 16.0404 ms (no swizzle)                    | ~8542           |
| 11              | Double Buffering         | 16.9039 ms                                 | ~8108           |
| -               | cuBLAS SGEMM             | 14.5 ms                                    | ~9462           |

> **Note:** GFLOP/s values are computed from \( \frac{2 \times N^3}{\text{runtime in seconds}} \).  
> cuBLAS achieves ~84% of theoretical peak here, and the best hand-tuned kernel reaches ~76%.
> **Note:** Timings represent the *best achieved* result for each kernel variant, not necessarily from the same code version. Parentheses provide additional notes.

---

## 📝 Summary of Learnings

This project was a deep dive into CUDA performance engineering—starting from a naive one-thread-per-output kernel (~615 ms) and iteratively refining it to a near-cuBLAS performer (~16 ms), achieving over a 38× speedup. The journey focused on exploiting the GPU memory hierarchy, optimizing execution mapping, and mitigating hardware bottlenecks.

Key themes explored:

- **Memory Access & Bandwidth Efficiency**  
  - Understanding and exploiting the CUDA memory hierarchy (registers, shared memory, L1/L2 caches, global memory).  
  - Achieving warp-wide global memory coalescing for both loads and stores.  
  - Using vectorized `float4` loads/stores for 128-bit transactions, with careful alignment/padding to handle non-multiple-of-4 dimensions.  

- **Data Reuse & Locality**  
  - Implementing shared memory tiling to amortize global memory latency.  
  - Moving from 1D to 2D block tiling for balanced reuse of rows and columns.  
  - Register tiling to reduce shared memory traffic.

- **Execution Mapping & Thread Organization**  
  - Transitioning from one-output-per-thread to multi-output-per-thread designs for better GPU utilization.  
  - Specializing warp-level work assignment (warp tiling) to align computation granularity with hardware execution units.  
  - Autotuning tile sizes and launch configurations to match the Quadro RTX 5000’s SM and register constraints.

- **Latency Hiding & Pipelining**  
  - Experimenting with double buffering to overlap global memory loads with computation, and analyzing its trade-offs on Turing architecture.  
  - Understanding when additional buffering hurts occupancy due to shared memory footprint.

- **Hardware-Aware Optimization**  
  - Identifying and mitigating shared memory bank conflicts through both padding and swizzling, with architecture-specific performance differences.  
  - Leveraging compiler hints (`__launch_bounds__`, `#pragma unroll`, `constexpr`) for register allocation, loop unrolling, and reduced branching overhead.  

For a detailed kernel-by-kernel breakdown—including diagrams, formulas, code annotations, experimental deviations from the Siboehm baseline, and future experiments —see the full notes:  
📄 [SGEMM Optimization Notes](https://docs.google.com/document/d/1K0kRn2RzdPTzVd_ZB9ktYOvlfTi4ZblQvi5NCOVj6kw/edit?tab=t.0)


