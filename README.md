# ThrustNet — GPU Neural Networks via CUDA Thrust Primitives

> Softmax classifier and 1-hidden-layer MLP trained entirely on GPU using NVIDIA Thrust — no custom CUDA kernels, no cuBLAS, no cuDNN.

---

## Overview

ThrustNet implements two neural network architectures from scratch on the GPU using only CUDA Thrust parallel primitives. The constraint: every mathematical operation — dot products, activation functions, gradient updates — must be expressed as a composition of Thrust transforms and reductions. No hand-written kernels. No black-box libraries.

This project demonstrates GPU computing fundamentals, parallel algorithm composition, and the honest engineering discipline of profiling and explaining your own performance bottlenecks.

---

## Technical Stack

| Layer | Technology |
|---|---|
| GPU Compute | NVIDIA CUDA + Thrust (C++17) |
| Parallel Primitives | `thrust::transform`, `thrust::reduce`, `thrust::transform_reduce`, `thrust::copy` |
| Build | `nvcc` + `g++`, GNU Make |
| Cluster Execution | SLURM on NCSA Delta (A100 GPUs) |
| Profiling | CUDA Events + `std::chrono` |

---

## Architecture

### Milestone 1 — Softmax Classifier
Multiclass logistic regression with stabilized softmax, cross-entropy loss, and mini-batch SGD.

```
x ──→ [W·x + b] ──→ softmax ──→ ŷ
```

**Results on blobs2d_3class.csv (150 samples, 200 epochs):**
- Final loss: 0.0072
- Accuracy: **99.33%**

### Milestone 2 — 1-Hidden-Layer MLP
Fully-connected network with ReLU activation and full backpropagation through both layers, implemented using Thrust primitives.

```
x ──→ [W1·x + b1] ──→ ReLU ──→ [W2·h + b2] ──→ softmax ──→ ŷ
```

**Results on blobs2d_3class.csv (150 samples, 200 epochs):**
- Final loss: 0.0058
- Accuracy: **100.00%** (+0.67% over softmax baseline)

---

## GPU Primitives Library (`thrust_nn.h`)

All operations are implemented as pure Thrust compositions with no custom kernels:

```cpp
// Dot product via fused transform-reduce
float dot(device_vector<float>& a, device_vector<float>& b);

// In-place SAXPY: y = y + α·x
void saxpy(device_vector<float>& y, device_vector<float>& x, float alpha);

// Numerically stable softmax (max-subtraction trick)
void softmax(device_vector<float>& logits, device_vector<float>& probs);

// Element-wise ReLU and sigmoid
void relu_inplace(device_vector<float>& v);
void sigmoid_inplace(device_vector<float>& v);

// Cross-entropy loss with epsilon clamp
float cross_entropy_one(device_vector<float>& probs, int y);
```

---

## Performance Profiling (Milestone 3)

Both models were timed using CUDA events (GPU) and `std::chrono` (CPU) across two dataset sizes.

| Dataset | Samples | Epochs | GPU Time | CPU Time | CPU Speedup |
|---|---|---|---|---|---|
| blobs2d_3class.csv | 150 | 200 | 12.97s | 2.47ms | **5,251×** |
| large_blobs2d_3class.csv | 3,000 | 50 | 75.67s | 14.02ms | **5,397×** |

### Why the CPU Won — Bottleneck Analysis

The results reveal three structural performance bottlenecks in the implementation:

**1. Low GPU Function Utilization**
Thrust vector operations run in the same wall-clock time regardless of vector size due to fixed kernel launch overhead. Operating on size-2 vectors (2 input features) pays A100-level launch cost for trivial work — the GPU's parallelism is never actually exercised.

**2. Excessive Implicit Data Transfers**
Scalar indexing into `device_vector` (e.g., `d_probs[c]`, `d_b[c]`, `d_y[s]`) triggers implicit device-to-host PCIe transfers. With hundreds of such accesses per sample per epoch across 200 epochs, cumulative transfer overhead dominates runtime.

**3. Ghost Batching**
The mini-batch loop performs per-sample weight updates sequentially rather than accumulating gradients across the batch and updating once. True batching would express the entire batch as a matrix multiply — the workload GPUs are designed for.

### Proposed Optimizations

| Bottleneck | Fix |
|---|---|
| Low utilization | Run on high-dimensional feature datasets where vector operations saturate GPU warps |
| Implicit transfers | Bulk-copy full vectors to host once per sample via `thrust::copy`; do all scalar arithmetic host-side |
| Ghost batching | Reformulate batch as matrix multiply, accumulate gradients, single weight update per batch |

---

## Build & Run

**Requirements:** CUDA 12+, `nvcc`, `g++` (C++17), SLURM (for cluster execution)

```bash
# Build all targets
make all

# Generate large synthetic dataset
./bin/gen_blobs3

# Run GPU models (Milestones 1 & 2)
./bin/main data/blobs2d_3class.csv 200

# Run CPU baseline (Milestone 3)
./bin/cpumain data/blobs2d_3class.csv 200
```

**SLURM (Delta cluster):**
```bash
sbatch thrustnn.slurm
```

---

## Repository Structure

```
thrust-nn/
├── code/
│   ├── src/
│   │   ├── main.cu          # GPU: Softmax + MLP training + CUDA event timing
│   │   ├── cpu_main.cpp     # CPU: Softmax baseline + chrono timing
│   │   └── gen_blobs3.cu    # Synthetic 3-class dataset generator
│   ├── include/
│   │   ├── thrust_nn.h      # GPU primitive library (dot, saxpy, softmax, ReLU...)
│   │   └── data_io.h        # CSV loading + dataset utilities
│   ├── data/
│   │   ├── blobs2d_3class.csv
│   │   └── large_blobs2d_3class.csv
│   └── Makefile
└── Kuchar_ThrustNN_Blobs.pdf   # Profiling report
```

---

## Key Engineering Decisions

**Why no cuBLAS?** The project constraint forces first-principles GPU programming — understanding how matrix operations decompose into parallel primitives rather than treating them as black boxes. This is the difference between using a tool and understanding one.

**Why stabilized softmax?** Naive `exp(x)` overflows for large logits. Subtracting the max before exponentiating is mathematically equivalent but numerically safe — a standard technique in production ML systems.

**Why honest profiling?** The CPU outperforming the GPU by 5,000× is a real result, not a failure. It surfaces exactly the kind of architectural mismatch that production GPU engineers diagnose — and the proposed fixes map directly to how frameworks like PyTorch and cuDNN actually solve this problem.

---

## Author

**Griffin Kuchar** — CS + Math + Economics, Texas Christian University  
AMD Engineering Intern (2x) | AI/ML Researcher: TCU × Children's Health Dallas  
[GitHub: @gkuchar](https://github.com/gkuchar)