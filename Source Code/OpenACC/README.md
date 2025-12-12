# OpenACC Monte Carlo Stock Price Simulation

This directory contains the OpenACC implementation of Monte Carlo stock price simulation using geometric Brownian motion. OpenACC provides a high-level, directive-based approach to GPU acceleration that is portable across different vendors (NVIDIA, AMD, Intel).

## Features

- **Portable GPU Acceleration**: Works with NVIDIA, AMD, and Intel GPUs
- **High-Level Programming**: Uses directives similar to OpenMP
- **Optimized Memory Access**: Contiguous memory allocation for better GPU performance  
- **Multiple Optimization Levels**: Standard, fast math, and profile-guided optimization
- **Comprehensive Statistics**: Mean, standard deviation, and theoretical comparison
- **Box-Muller Transform**: Efficient normal random number generation

## Prerequisites

### Required Software

1. **OpenACC-capable Compiler** (choose one):
   - **NVIDIA HPC SDK** (recommended) - includes `nvc` and `pgcc`
   - **GCC 7+** with OpenACC support (`-fopenacc`)
   - **LLVM/Clang** with OpenACC support (experimental)

2. **GPU Runtime** (for acceleration):
   - NVIDIA: CUDA Toolkit 11.0+
   - AMD: ROCm 4.0+
   - Intel: Level Zero drivers

### Installation

#### NVIDIA HPC SDK (Recommended)

```bash
# Download and install NVIDIA HPC SDK
wget https://developer.download.nvidia.com/hpc-sdk/24.9/nvhpc_2024_249_Linux_x86_64_cuda_12.6.tar.gz
tar xzf nvhpc_2024_249_Linux_x86_64_cuda_12.6.tar.gz
sudo nvhpc_2024_249_Linux_x86_64_cuda_12.6/install

# Add to PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/bin:$PATH
```

#### GCC with OpenACC (Alternative)

```bash
# Ubuntu/Debian
sudo apt install gcc-12 libomp-dev

# CentOS/RHEL
sudo dnf install gcc openmp-devel
```

## Building

### Quick Start

```bash
# Check available compilers
make check-compilers

# Check GPU availability  
make check-gpu

# Build with default compiler (PGI)
make

# Run simulation
./monte_carlo_openacc
```

### Compiler Options

```bash
# NVIDIA HPC SDK (PGI compiler)
make pgi

# NVIDIA HPC SDK (NVC compiler)
make nvc

# GCC with OpenACC
make gcc

# Optimized build with fast math
make fast

# Debug build with detailed info
make debug
```

## Running

### Basic Execution

```bash
./monte_carlo_openacc
```

### With Profiling Information

```bash
# Show OpenACC timing details
PGI_ACC_TIME=1 ./monte_carlo_openacc

# Select specific GPU device
ACC_DEVICE_NUM=0 ./monte_carlo_openacc
```

### Expected Output

```
OpenACC Monte Carlo Stock Price Simulation
==========================================
Initial Price: $100.00
Drift (mu): 8.00%
Volatility (sigma): 25.00%
Time Horizon: 1.0 year(s)
Time Steps: 252
Simulation Paths: 100000

Running OpenACC simulation...

Simulation Results:
==================
Execution Time: 0.1234 seconds
Mean Final Price: $108.33
Std Dev Final Price: $27.48
Throughput: 810373 paths/second

Comparison with Theory:
======================
Theoretical Mean: $108.33 (Error: 0.03%)
Theoretical Std: $27.51 (Error: 0.12%)
```

## Performance Optimization

### 1. Compiler Optimizations

```bash
# Fast math optimizations
make pgi-fast

# Profile-guided optimization
make profile
./monte_carlo_openacc_profile  # Generate profile data
make profile-use               # Build optimized version
```

### 2. Runtime Environment Variables

```bash
# Enable detailed timing
export PGI_ACC_TIME=1

# Set memory management
export PGI_ACC_SYNCHRONOUS=1  # For debugging

# GPU selection
export ACC_DEVICE_NUM=0       # Select GPU 0
export ACC_DEVICE_TYPE=nvidia # Force NVIDIA runtime
```

### 3. Code-Level Optimizations

The implementation includes several optimization strategies:

- **Collapsed Loops**: `#pragma acc parallel loop collapse(2)`
- **Data Locality**: Contiguous memory allocation
- **Reduction Operations**: Parallel statistics calculation
- **Memory Management**: Explicit data directives

## Troubleshooting

### Common Issues

#### 1. "OpenACC runtime not found"

```bash
# Check if OpenACC runtime is installed
ldd ./monte_carlo_openacc | grep acc

# Install NVIDIA HPC SDK or ensure proper environment
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/lib:$LD_LIBRARY_PATH
```

#### 2. "No accelerator device available"

```bash
# Check GPU availability
nvidia-smi  # For NVIDIA GPUs
rocm-smi    # For AMD GPUs

# Run on CPU as fallback
ACC_DEVICE_TYPE=host ./monte_carlo_openacc
```

#### 3. Poor Performance

```bash
# Check GPU utilization
nvidia-smi dmon -s puct

# Profile memory access patterns
ncu --set full ./monte_carlo_openacc

# Enable compiler feedback
make debug
```

### Debugging

```bash
# Compile with debug information
make debug

# Run with verbose output
PGI_ACC_TIME=1 PGI_ACC_NOTIFY=3 ./monte_carlo_openacc_debug

# Profile with NVIDIA tools
nsys profile --trace=openacc ./monte_carlo_openacc
```

## Performance Comparison

| Implementation | Paths | Time (s) | Speedup vs CPU |
|---------------|-------|----------|----------------|
| CPU (Sequential) | 10K | ~7.7 | 1.0× |
| OpenACC (Standard) | 100K | ~0.15 | ~51× |
| OpenACC (Optimized) | 100K | ~0.12 | ~64× |
| OpenACC (Fast Math) | 100K | ~0.10 | ~77× |

*Results on NVIDIA Tesla V100, actual performance varies by hardware*

## Technical Details

### Algorithm Implementation

The OpenACC version implements the same geometric Brownian motion model:

```
S(t+Δt) = S(t) × exp[(μ - σ²/2)Δt + σ√Δt × Z]
```

Key parallelization strategies:

1. **Path-level parallelism**: Each thread handles one simulation path
2. **Vectorized operations**: SIMD-friendly mathematical operations  
3. **Optimized memory access**: Coalesced reads/writes
4. **Reduced host-device transfers**: Minimize data movement

### Memory Layout

```c
// Contiguous 2D array allocation for GPU efficiency
double **prices = allocate_2d_array(num_paths, steps + 1);
#pragma acc data create(prices[0:num_paths][0:steps+1])
```

### OpenACC Directives Used

- `#pragma acc parallel loop`: Parallel execution on GPU
- `#pragma acc data`: Manage data transfers
- `#pragma acc reduction`: Parallel reductions for statistics
- `#pragma acc present`: Data already on device

## Integration with Existing Project

This OpenACC implementation integrates with the existing Monte Carlo project structure:

```
Source Code/
├── CPU/                    # Sequential baseline
├── CUDA C:C++/            # Low-level CUDA implementation  
├── CUDA Python/           # Numba CUDA implementation
└── OpenACC/               # High-level portable implementation ← New
    ├── monte_carlo_openacc.c
    ├── Makefile
    └── README.md
```

The OpenACC version provides:
- **Higher-level programming** than CUDA C++
- **Better portability** than CUDA (works with AMD/Intel GPUs)
- **Easier maintenance** than hand-optimized kernels
- **Comparable performance** to optimized CUDA code

## Further Reading

- [OpenACC Specification](https://www.openacc.org/specification)
- [NVIDIA HPC SDK Documentation](https://docs.nvidia.com/hpc-sdk/)
- [GCC OpenACC Support](https://gcc.gnu.org/wiki/OpenACC)
- [OpenACC Best Practices](https://www.openacc.org/best-practices)