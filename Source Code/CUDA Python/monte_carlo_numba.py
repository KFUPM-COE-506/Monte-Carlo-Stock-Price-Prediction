"""
CUDA Python/Numba Implementation for Monte Carlo Stock Price Simulation

This module provides GPU-accelerated Monte Carlo simulation using Numba CUDA
for financial derivative pricing using geometric Brownian motion.

Course: 251-COE-506-01 (GPU Programming & Architecture)
Institution: King Fahd University of Petroleum & Minerals
"""

import numpy as np
import time
from numba import cuda
import math


@cuda.jit
def monte_carlo_kernel_numba(prices, S0, mu, sigma, dt, steps, num_paths, seed):
    """
    CUDA kernel for Monte Carlo simulation using Numba.
    Each thread simulates one complete price path.

    Key Performance Areas:
    - Random number generation (RNG)
    - Exponential calculations
    - Memory access patterns

    Parameters
    ----------
    prices : device array
        Output array for storing price trajectories
    S0 : float
        Initial stock price
    mu : float
        Drift coefficient
    sigma : float
        Volatility coefficient
    dt : float
        Time step size
    steps : int
        Number of time steps
    num_paths : int
        Number of simulation paths
    seed : int
        Random seed for RNG
    """
    path_id = cuda.grid(1)

    if path_id < num_paths:
        # Initialize random state for this thread
        # Using simple LCG (Linear Congruential Generator) for demonstration
        # In production, use cuRAND for better quality
        rng_state = seed + path_id

        # Set initial price
        price = S0
        prices[path_id * (steps + 1)] = price

        # Simulate price path through time
        for step in range(1, steps + 1):
            # BOTTLENECK #1: Random Number Generation
            # Generate two uniform random numbers
            rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
            u1 = (rng_state & 0xFFFFFF) / 16777216.0

            rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
            u2 = (rng_state & 0xFFFFFF) / 16777216.0

            # Box-Muller transform to get normal distribution
            z = math.sqrt(-2.0 * math.log(u1 + 1e-10)) * math.cos(2.0 * math.pi * u2)

            # Calculate price components
            drift = (mu - 0.5 * sigma * sigma) * dt
            diffusion = sigma * math.sqrt(dt) * z

            # BOTTLENECK #2: Exponential Calculation
            # Update price using geometric Brownian motion
            price = price * math.exp(drift + diffusion)

            # BOTTLENECK #3: Memory Access Pattern
            # Store result (potentially uncoalesced access)
            prices[path_id * (steps + 1) + step] = price


def run_monte_carlo_gpu_numba(S0, mu, sigma, T, steps, num_paths):
    """
    Wrapper function to run GPU Monte Carlo simulation with Numba.

    Parameters
    ----------
    S0 : float
        Initial stock price
    mu : float
        Drift coefficient (expected return)
    sigma : float
        Volatility coefficient
    T : float
        Time horizon in years
    steps : int
        Number of time discretization steps
    num_paths : int
        Number of simulation paths

    Returns
    -------
    prices : ndarray
        Simulated prices on host memory, shape (num_paths, steps+1)
    exec_time : float
        Kernel execution time (excluding memory transfers)
    """
    dt = T / steps

    # Allocate device memory
    prices_device = cuda.device_array(num_paths * (steps + 1), dtype=np.float32)

    # Configure kernel launch parameters
    threads_per_block = 256
    blocks_per_grid = (num_paths + threads_per_block - 1) // threads_per_block

    print(f"Kernel Configuration:")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Blocks per grid: {blocks_per_grid}")
    print(f"  Total threads: {threads_per_block * blocks_per_grid:,}")

    # Launch kernel and measure execution time
    cuda.synchronize()
    kernel_start = time.time()

    monte_carlo_kernel_numba[blocks_per_grid, threads_per_block](
        prices_device, S0, mu, sigma, dt, steps, num_paths, 12345
    )

    cuda.synchronize()
    kernel_time = time.time() - kernel_start

    # Copy results back to host
    prices_host = prices_device.copy_to_host()
    prices_host = prices_host.reshape(num_paths, steps + 1)

    return prices_host, kernel_time


if __name__ == "__main__":
    # Example usage
    if not cuda.is_available():
        print("CUDA is not available. Please ensure you have a CUDA-capable GPU.")
        exit(1)

    # Simulation parameters
    S0 = 100.0      # Initial stock price
    mu = 0.08       # Drift (8% annual)
    sigma = 0.25    # Volatility (25% annual)
    T = 1.0         # Time horizon (1 year)
    steps = 252     # Time steps (trading days)
    num_paths = 100000  # Number of simulation paths

    print("Running GPU Monte Carlo Simulation (Numba CUDA)...")
    print(f"Paths: {num_paths:,}, Steps: {steps}\n")

    start_time = time.time()
    prices, kernel_time = run_monte_carlo_gpu_numba(S0, mu, sigma, T, steps, num_paths)
    total_time = time.time() - start_time

    # Calculate statistics
    final_prices = prices[:, -1]
    mean_price = np.mean(final_prices)
    std_price = np.std(final_prices)

    print(f"\nTotal Time (including transfers): {total_time:.4f} seconds")
    print(f"Kernel Execution Time: {kernel_time:.4f} seconds")
    print(f"Memory Transfer Time: {total_time - kernel_time:.4f} seconds")
    print(f"Mean Final Price: ${mean_price:.2f}")
    print(f"Std Dev Final Price: ${std_price:.2f}")
    print(f"Throughput: {num_paths/total_time:,.0f} paths/second")


