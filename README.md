# MetalNet - High-Performance C++ CNN Library

**Header-only, C++20, zero-dependency**

Documentation: https://kunwarprabhat.github.io/MetalNet_Documentation/

---

## Overview

MetalNet is a header-only Convolutional Neural Network library implemented in pure C++20. It achieves CPU performance that matches or exceeds mainstream frameworks by using:

* Zero-overhead abstractions
* Directed Acyclic Graph (DAG) based autograd
* Implicit GEMM convolution (no memory explosion)
* AVX2 and FMA intrinsics
* OpenMP multithreading with adaptive thread control
* L1/L2 cache-aware tiling

No external dependencies beyond a C++20 compiler with OpenMP and AVX2 support.

---

## Documentation

All detailed explanations, API references, optimization deep-dives, and benchmark comparisons are available at:

**[MetalNet Documentation](https://kunwarprabhat.github.io/MetalNet_Documentation/)**

The documentation covers:

* Architecture and design philosophy
* Step-by-step usage examples
* Compilation flags and performance tuning
* Problems encountered and their solutions
* Benchmarks vs PyTorch (MNIST, CIFAR, deep CNNs)

---

## Getting Started

### Prerequisites

* C++20 compiler (GCC 10+, Clang 14+, or MSVC 2022 with `/std:c++20`)
* CPU with AVX2 and FMA support (Intel Haswell or later, AMD Excavator or later)
* OpenMP library (optional, for multithreading)

---

### Installation

Clone the repository:

```bash
git clone https://github.com/KunwarPrabhat/CustomCNN.git
cd CustomCNN
```

Include the header in your project:

```cpp
#include "MetalNet/MetalNet.h"
```

No build or linking step is required – it is header-only.

---

## Minimal Example

```cpp
#include "MetalNet/MetalNet.h"
using namespace MetalNet;

int main() {
    // Build a simple CNN
    Model model;
    model << conv2d(1, 32, 3, 1, 1) << relu()
          << maxpool2d(2, 2)
          << conv2d(32, 64, 3, 1, 1) << relu()
          << flatten()
          << dense(64 * 14 * 14, 10);

    // Compile with input shape (batch, channels, height, width)
    model.compile({32, 1, 28, 28});

    // Forward pass
    Tensor input(32, 1, 28, 28);
    Tensor output = model.forward(input);

    // Switch to inference mode (releases gradient buffers)
    model.eval();
    return 0;
}
```

For training, use `model.backward(loss_gradient)` and the built-in optimizers (SGD, Adam). See the documentation for full training examples.

---

## Compilation Flags (Performance)

For best performance, compile with:

```bash
g++ -std=c++20 -O3 -mavx2 -mfma -fopenmp -march=native -ffast-math -I./include main.cpp -o my_program
```

| Flag            | Purpose                               |
| --------------- | ------------------------------------- |
| `-mavx2 -mfma`  | Enable AVX2 and FMA SIMD instructions |
| `-fopenmp`      | Enable multithreading                 |
| `-march=native` | Optimize for host CPU                 |
| `-ffast-math`   | Improves SIMD performance             |
| `-O3`           | Maximum optimization                  |

---

### Windows (MSVC)

```bash
cl /std:c++20 /O2 /arch:AVX2 /openmp /Iinclude main.cpp
```

---

## Key Optimizations

* **Implicit GEMM** – Convolution executed as matrix multiplication without materializing im2col buffer
* **Zero-Allocation Hot Path** – All buffers pre-allocated during `compile()`
* **AVX2 Register Accumulation** – `_mm256_fmadd_ps` keeps values in registers
* **Cache Tiling** – Loop blocking optimized for L2 cache (`BLOCK_SIZE = 64`)
* **Adaptive Multithreading** – OpenMP disabled for small workloads
* **Custom NoInitAllocator** – Avoids zero-initialization overhead

---

## Benchmarks

MetalNet consistently outperforms PyTorch CPU backend on VGG-style models:

* Up to **2.3× faster inference** on 8-core CPU
* **Optimized memory usage** for deep CNNs
* No memory duplication (im2col avoided)

Detailed benchmarks are available in the documentation and `benchmarks/` folder.

---

## Repository Structure

```
CustomCNN/
├── include/
│   └── MetalNet/
│       ├── MetalNet.h
│       ├── tensor.h
│       ├── layer.h
│       ├── conv2d.h
│       ├── pooling.h
│       ├── activation.h
│       ├── loss.h
│       └── ...
├── benchmarks/
├── examples/
├── tests/
└── README.md
```

---

## Licence 
BSD 3-Clause License - see LICENSE file.

---

## Author

**Kunwar Prabhat**
GitHub: https://github.com/KunwarPrabhat
