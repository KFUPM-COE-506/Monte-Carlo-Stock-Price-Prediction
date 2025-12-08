# Monte Carlo Stock Price Simulation (CPU vs GPU Acceleration)

**Course:** 251-COE-506-01 (GPU Programming & Architecture)
**Institution:** King Fahd University of Petroleum & Minerals
**Semester:** Fall 2025

**Team Members:**
Mahdi Al Mayuf, Khaleel Alhaboub, Haitham Alsaeed

---

## Executive Summary

This comprehensive analysis presents a systematic investigation of GPU acceleration techniques applied to Monte Carlo simulation for financial derivative pricing. Through rigorous profiling using NVIDIA's nsys toolkit, we identify critical performance bottlenecks and propose theoretically-grounded optimization strategies. Our baseline implementation achieves a **55.9× speedup** over sequential CPU execution, with projected improvements reaching **150-180× speedup** through targeted optimizations.

**Key Contributions:**

- Quantitative performance characterization of parallel Monte Carlo simulation
- Systematic identification of computational bottlenecks via hardware performance counters
- Evidence-based optimization recommendations with projected performance gains
- Comparative analysis of GPU programming paradigms (Numba CUDA vs. CUDA C/C++)

---

## 1. Introduction and Motivation

### 1.1 Financial Computing Context

Monte Carlo simulation represents a cornerstone methodology in computational finance, particularly for pricing complex derivatives and conducting risk analysis under uncertainty. The geometric Brownian motion (GBM) model, described by the stochastic differential equation:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

requires extensive numerical integration over millions of simulated price paths to achieve statistical significance. Traditional CPU-based implementations face severe computational constraints, limiting real-time applicability in high-frequency trading and dynamic risk management scenarios.

### 1.2 GPU Acceleration Rationale

The inherent parallelism in Monte Carlo methods—where each simulation path is independent—makes them ideally suited for GPU acceleration. Modern GPUs provide:

- **Massive parallelism:** 1000s of concurrent threads
- **High memory bandwidth:** >500 GB/s on modern architectures
- **Specialized compute units:** Tensor cores and optimized floating-point pipelines

### 1.3 Research Objectives

This study systematically addresses the following objectives:

1. **Baseline Characterization:** Establish performance metrics for CPU and GPU implementations
2. **Bottleneck Identification:** Utilize hardware performance counters to identify computational hotspots
3. **Optimization Strategy:** Develop evidence-based optimization recommendations
4. **Validation:** Verify numerical accuracy and performance improvements

---

## 2. Methodology

### 2.1 Experimental Framework

Our analysis employs a multi-tiered approach:

**Platform:** Google Colab with NVIDIA Tesla T4 GPU (Turing architecture, SM 7.5)

**Profiling Tools:**

- `nsys`: Command-line profiler for kernel-level performance metrics
- Hardware performance counters: Occupancy, memory efficiency, instruction throughput

**Implementation Variants:**

1. **CPU Baseline:** NumPy-based sequential implementation
2. **Numba CUDA:** Python-based GPU acceleration with JIT compilation
3. **CUDA C/C++:** Low-level optimized kernel implementation

### 2.2 Performance Metrics

We evaluate performance across multiple dimensions:

- **Execution Time:** Wall-clock time and kernel-specific timing
- **Throughput:** Simulations per second
- **GPU Utilization:** Achieved occupancy, SM efficiency
- **Memory Performance:** Bandwidth utilization, coalescing efficiency
- **Computational Efficiency:** FLOPS, instruction-level parallelism

### Notebook Structure

The notebook is organized into sections:

1. **Environment Configuration**: GPU verification and dependency installation
2. **Simulation Parameters**: Financial model parameters
3. **CPU Baseline**: Sequential implementation and execution
4. **GPU Numba**: Python-based GPU acceleration
5. **CUDA C/C++**: Low-level CUDA implementation
6. **Profiling**: nsys-based performance analysis
7. **Visualization**: Performance and statistical plots
8. **Analysis**: Bottleneck identification and optimization strategies

---

## Installation Instructions

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (compute capability 7.0+)
- **CUDA Toolkit**: Version 11.0 or higher
- **Python**: Version 3.8 or higher
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS (with CUDA support)

### Step 1: Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU availability
nvidia-smi

# Verify CUDA runtime
python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"
```

### Step 2: Install Python Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or manually
pip install numpy>=1.21.0
pip install numba>=0.55.0
pip install matplotlib>=3.5.0
pip install pandas>=1.3.0
pip install jupyter  # For notebook execution
```

### Step 3: Install NVIDIA Profiling Tools (Optional)

For profiling analysis, install NVIDIA Nsight Systems:

**Ubuntu/Debian:**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb
sudo apt install ./nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb
```

**Or use the installation script in the notebook** (Cell 4) for Google Colab.

---

## Running the Notebook

### Google Colab (Recommended)

1. **Upload Notebook**

   - Click "File" > "Upload notebook"
   - Select `MonteCarlo_GPU_Profiling.ipynb`

### Local Jupyter Environment

1. **Install Jupyter**

   ```bash
   pip install jupyter
   ```

2. **Launch Jupyter**

   ```bash
   jupyter notebook MonteCarlo_GPU_Profiling.ipynb
   ```

3. **Run Cells**
   - Execute cells sequentially
   - Ensure GPU is available: `nvidia-smi` should show your GPU

## Running Standalone Code

### CPU Baseline Implementation

**Location**: `Source Code/CPU/monte_carlo_cpu.py`

**Run directly:**

```bash
cd "Source Code/CPU"
python monte_carlo_cpu.py
```

**Use as module:**

```python
import sys
sys.path.append('Source Code/CPU')
from monte_carlo_cpu import monte_carlo_cpu
import numpy as np

prices = monte_carlo_cpu(S0=100.0, mu=0.08, sigma=0.25, T=1.0, steps=252, num_paths=10000)
```

**Expected Output:**

- Execution time: ~7-8 seconds for 10,000 paths
- Mean final price: ~$108.33
- Standard deviation: ~$27.50

### CUDA Python (Numba) Implementation

**Location**: `Source Code/CUDA Python/monte_carlo_numba.py`

**Run directly:**

```bash
cd "Source Code/CUDA Python"
python monte_carlo_numba.py
```

**Use as module:**

```python
import sys
sys.path.append('Source Code/CUDA Python')
from monte_carlo_numba import run_monte_carlo_gpu_numba
import numpy as np

prices, kernel_time = run_monte_carlo_gpu_numba(
    S0=100.0, mu=0.08, sigma=0.25, T=1.0, steps=252, num_paths=100000
)
```

**Expected Output:**

- Total time: ~1.5 seconds for 100,000 paths
- Kernel time: ~1.2 seconds
- Speedup: ~50× vs CPU

### CUDA C/C++ Implementation

**Location**: `Source Code/CUDA C:C++/`

**Compile and run:**

```bash
cd "Source Code/CUDA C:C++"

# Compile
make

# Run
./monte_carlo_main

# Or compile manually
nvcc -O3 -arch=sm_75 monte_carlo_main.cu -o monte_carlo_main -lcurand
```

**Expected Output:**

- Execution time: ~0.065 seconds for 1,000,000 paths
- Mean final price: ~$108.33
- Speedup: ~55.9× vs CPU

**Compile optimized version:**

```bash
make monte_carlo_main-fast  # Uses -use_fast_math flag
```

**Compile as shared library (for Python binding):**

```bash
make monte_carlo.so
```

---

## Profiling Guide

### Using nsys (NVIDIA Nsight Systems)

**Basic profiling:**

```bash
nsys profile --stats=true ./monte_carlo_main
```

**Detailed profiling with metrics:**

```bash
nsys profile --stats=true \
    --trace=cuda,nvtx,osrt \
    -o profile_output \
    ./monte_carlo_main
```

**View results:**

```bash
# Command-line summary
nsys stats profile_output.nsys-rep

# GUI visualization (if nsys-ui installed)
nsys-ui profile_output.nsys-rep
```

### Interpreting Results

**Good Performance Indicators:**

- Achieved occupancy >70%
- Memory efficiency >80%
- Kernel time dominates (not memory transfers)
- High SM efficiency

**Bottleneck Indicators:**

- Low memory efficiency (<70%) -> Uncoalesced access
- Low occupancy (<50%) -> Resource underutilization
- High instruction count -> Optimization opportunities

---

## Expected Outputs

### Performance Metrics

| Implementation | Paths | Time (s) | Speedup |
| -------------- | ----- | -------- | ------- |
| CPU Baseline   | 10K   | ~7.7     | 1.0×    |
| GPU Numba      | 100K  | ~1.5     | ~51×    |
| GPU CUDA C++   | 1M    | ~0.065   | ~55.9×  |

### Statistical Results

For standard parameters (S₀=$100, μ=8%, σ=25%, T=1yr):

- **Theoretical Mean**: $108.33
- **Theoretical Std Dev**: $27.51
- **Simulated Mean**: $108.06 - $108.33 (error <0.5%)
- **Simulated Std Dev**: $27.14 - $27.52 (error <2%)

### Generated Files

**From Notebook:**

- `profiling_output.txt`: Complete nsys output
- `*.nsys-rep`, `*.sqlite`: Profiling data files
- `sample_paths.png`: Sample simulation paths visualization
- `price_distribution.png`: Final price distribution histogram
- `performance_comparison.png`: CPU vs GPU comparison
- `bottleneck_breakdown.png`: Bottleneck analysis pie chart
- `optimization_impact.png`: Projected optimization improvements

#### Thanks for reading.
