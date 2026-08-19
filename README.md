cuBLAS Demo — BLAS-level GPU Acceleration

![status](https://img.shields.io/badge/status-scaffold-ready-yellow)

Demonstrates how to integrate cuBLAS into C++ projects for high-performance linear algebra. Includes CPU fallback code, build instructions, and notes for performance tuning.

Highlights
- CPU fallback with explanatory comments
- Guidance for integrating cuBLAS functions (saxpy, dot, gemm)
- CMake build and notes for linking NVIDIA libraries

Quickstart

```powershell
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
cmake --build .
```

Demo GIF placeholder:

![cublas-demo](./assets/cublas_demo.gif)

Mermaid diagram

```mermaid
flowchart LR
	A[Host arrays] --> B[cuBLAS API]
	B --> C[Device computation]
	C --> D[Host results]
```

Why this impresses
- cuBLAS is a core building block for accelerated DL ops; recruiters expect familiarity

License: MIT

