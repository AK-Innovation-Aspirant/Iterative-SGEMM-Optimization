# Iterative SGEMM Optimization

This project documents the iterative optimization of FP32 SGEMM (Single-Precision General Matrix Multiplication) for large square matrices (4092x4092) using CUDA. It follows the progression based on the highly cited Siboehm Blog (link below). The goal was to learn the fundamentals of low-level CUDA programming and extend it to a real-world problem. This was a personal side-project done during my internship **Intel Labs** with the GPU used being a **QUADRO RTX 5000**.

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

## 📝 Summary of Learnings

A detailed summary of my learnings and performance tuning journey(and experiments in addition to the Siboehm github implementation) is available here:

📄 [SGEMM Optimization Notes (Google Doc)](https://docs.google.com/document/d/1K0kRn2RzdPTzVd_ZB9ktYOvlfTi4ZblQvi5NCOVj6kw/edit?tab=t.0)

This includes, but is not limited to:

- Memory hierarchy insights and register usage  
- Shared memory tiling strategies  
- Thread/block tuning and occupancy considerations  
- Performance bottlenecks and warp-level optimizations


