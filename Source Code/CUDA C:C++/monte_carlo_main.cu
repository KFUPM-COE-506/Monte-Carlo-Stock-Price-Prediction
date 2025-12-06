/*
 * Complete CUDA C/C++ Program for Monte Carlo Stock Price Simulation
 *
 * This is a standalone executable program that demonstrates the Monte Carlo
 * simulation with comprehensive profiling support.
 *
 * Course: 251-COE-506-01 (GPU Programming & Architecture)
 * Institution: King Fahd University of Petroleum & Minerals
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

/**
 * CUDA kernel for Monte Carlo stock price simulation.
 * Each thread simulates one complete price path.
 */
__global__ void monte_carlo_kernel(
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
            // Called millions of times in total
            price = price * expf(drift + diffusion);

            // BOTTLENECK #3: Uncoalesced Memory Access
            // Access pattern: prices[path_id * (steps+1) + step]
            // Consecutive threads access memory with stride of (steps+1)
            prices[path_id * (steps + 1) + step] = price;
        }
    }
}

int main() {
    // Simulation parameters
    float S0 = 100.0f;
    float mu = 0.08f;
    float sigma = 0.25f;
    float T = 1.0f;
    int steps = 252;
    int num_paths = 1000000;

    printf("Monte Carlo Stock Price Simulation\n");
    printf("===================================\n");
    printf("Initial Price: $%.2f\n", S0);
    printf("Drift (mu): %.2f%%\n", mu * 100);
    printf("Volatility (sigma): %.2f%%\n", sigma * 100);
    printf("Time Horizon: %.1f year\n", T);
    printf("Time Steps: %d\n", steps);
    printf("Number of Paths: %d\n", num_paths);
    printf("Total Calculations: %ld\n\n", (long)num_paths * steps);

    float dt = T / steps;
    size_t size = num_paths * (steps + 1) * sizeof(float);

    // Allocate device memory
    float* d_prices;
    cudaError_t err = cudaMalloc(&d_prices, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Configure kernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;

    printf("Kernel Configuration:\n");
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Blocks per grid: %d\n", blocks_per_grid);
    printf("  Total threads: %d\n\n", threads_per_block * blocks_per_grid);

    // Warm-up run
    printf("Running warm-up...\n");
    monte_carlo_kernel<<<100, threads_per_block>>>(
        d_prices, S0, mu, sigma, dt, steps, 10000, 12345ULL
    );
    cudaDeviceSynchronize();

    // Main profiled run
    printf("Starting main simulation (this will be profiled)...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    monte_carlo_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_prices, S0, mu, sigma, dt, steps, num_paths, 12345ULL
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_prices);
        return 1;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back (sample only for verification)
    float* h_prices = (float*)malloc(num_paths * (steps + 1) * sizeof(float));
    if (h_prices == NULL) {
        fprintf(stderr, "Host malloc failed\n");
        cudaFree(d_prices);
        return 1;
    }

    err = cudaMemcpy(h_prices, d_prices, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed: %s\n", cudaGetErrorString(err));
        free(h_prices);
        cudaFree(d_prices);
        return 1;
    }

    // Calculate statistics
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < num_paths; i++) {
        float final_price = h_prices[i * (steps + 1) + steps];
        sum += final_price;
        sum_sq += final_price * final_price;
    }
    double mean = sum / num_paths;
    double variance = (sum_sq / num_paths) - (mean * mean);
    double std_dev = sqrt(variance);

    printf("\n=== Results ===\n");
    printf("Kernel Execution Time: %.4f seconds\n", milliseconds / 1000.0);
    printf("Mean Final Price: $%.2f\n", mean);
    printf("Std Dev Final Price: $%.2f\n", std_dev);
    printf("95%% Confidence Interval: [$%.2f, $%.2f]\n",
           mean - 1.96 * std_dev, mean + 1.96 * std_dev);

    // Cleanup
    cudaFree(d_prices);
    free(h_prices);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


