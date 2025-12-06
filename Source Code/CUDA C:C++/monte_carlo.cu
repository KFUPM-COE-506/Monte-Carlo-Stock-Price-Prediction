/*
 * CUDA C/C++ Kernel Implementation for Monte Carlo Stock Price Simulation
 *
 * This file contains the CUDA kernel for GPU-accelerated Monte Carlo simulation
 * for financial derivative pricing using geometric Brownian motion.
 *
 * Course: 251-COE-506-01 (GPU Programming & Architecture)
 * Institution: King Fahd University of Petroleum & Minerals
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

/**
 * CUDA kernel for Monte Carlo stock price simulation.
 * Each thread simulates one complete price path.
 *
 * @param prices Output array for storing price trajectories
 * @param S0 Initial stock price
 * @param mu Drift coefficient (expected return)
 * @param sigma Volatility coefficient
 * @param dt Time step size
 * @param steps Number of time steps
 * @param num_paths Number of simulation paths
 * @param seed Random seed for cuRAND
 */
__global__ void monte_carlo_kernel_cuda(
    float* prices,
    float S0,
    float mu,
    float sigma,
    float dt,
    int steps,
    int num_paths,
    unsigned long long seed
) {
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_id < num_paths) {
        // BOTTLENECK #1: cuRAND State Initialization
        // Each thread initializes its own RNG state
        // This is expensive and happens for every thread
        curandState state;
        curand_init(seed, path_id, 0, &state);

        // Set initial price
        float price = S0;
        prices[path_id * (steps + 1)] = price;

        // Simulate price path
        for (int step = 1; step <= steps; step++) {
            // BOTTLENECK #1 (continued): Random Number Generation
            // curand_normal() is called millions of times
            float z = curand_normal(&state);

            // Calculate drift and diffusion components
            float drift = (mu - 0.5f * sigma * sigma) * dt;
            float diffusion = sigma * sqrtf(dt) * z;

            // BOTTLENECK #2: Exponential Function
            // expf() is a transcendental function with high latency (23-30 cycles)
            // Called 252 million times for 1M paths
            price = price * expf(drift + diffusion);

            // BOTTLENECK #3: Uncoalesced Memory Access
            // Access pattern: prices[path_id * (steps+1) + step]
            // Consecutive threads access memory with stride of (steps+1)
            // This reduces memory throughput
            prices[path_id * (steps + 1) + step] = price;
        }
    }
}

/**
 * C interface wrapper for launching the Monte Carlo kernel.
 * This function can be called from Python or other languages.
 */
extern "C" {
    void launch_monte_carlo(
        float* d_prices,
        float S0,
        float mu,
        float sigma,
        float T,
        int steps,
        int num_paths
    ) {
        float dt = T / steps;
        int threads_per_block = 256;
        int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;

        monte_carlo_kernel_cuda<<<blocks_per_grid, threads_per_block>>>(
            d_prices, S0, mu, sigma, dt, steps, num_paths, 12345ULL
        );

        cudaDeviceSynchronize();
    }
}


