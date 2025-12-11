/*
 * Advanced OpenACC Implementation with Multiple Optimization Strategies
 * 
 * This module demonstrates advanced OpenACC techniques for Monte Carlo simulation
 * including async operations, multiple GPU support, and various optimization patterns.
 * 
 * Course: 251-COE-506-01 (GPU Programming & Architecture)
 * Institution: King Fahd University of Petroleum & Minerals
 * 
 * Compilation:
 *   nvc -acc -Minfo=accel -fast monte_carlo_advanced.c -lm -o monte_carlo_advanced
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Configuration constants
#define MAX_PATHS 1000000
#define MAX_STEPS 1000
#define BLOCK_SIZE 256

/**
 * Advanced random number generator using Linear Congruential Generator
 * Optimized for parallel execution on GPU
 */
typedef struct {
    unsigned long long seed;
    unsigned long long a;
    unsigned long long c;
} rng_state_t;

void init_rng(rng_state_t *state, unsigned long long seed) {
    state->seed = seed;
    state->a = 1664525ULL;
    state->c = 1013904223ULL;
}

double rng_uniform(rng_state_t *state) {
    state->seed = (state->a * state->seed + state->c);
    return (double)(state->seed & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

double rng_normal(rng_state_t *state) {
    static int has_spare = 0;
    static double spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    
    has_spare = 1;
    double u = rng_uniform(state);
    double v = rng_uniform(state);
    double mag = sqrt(-2.0 * log(u));
    spare = mag * cos(2.0 * M_PI * v);
    return mag * sin(2.0 * M_PI * v);
}

/**
 * Kernel-level Monte Carlo simulation with advanced OpenACC optimizations
 * 
 * Features:
 * - Vector-friendly operations
 * - Cache-optimized memory access
 * - Gang, worker, vector parallelism
 * - Async operations
 */
void monte_carlo_kernel_optimized(double S0, double mu, double sigma, double T,
                                 int steps, int num_paths, double *final_prices,
                                 unsigned long long *seeds) {
    
    double dt = T / steps;
    double drift_coeff = (mu - 0.5 * sigma * sigma) * dt;
    double diffusion_coeff = sigma * sqrt(dt);
    
    #pragma acc parallel loop gang worker vector \
                             present(final_prices[0:num_paths], seeds[0:num_paths]) \
                             private(drift_coeff, diffusion_coeff)
    for (int path = 0; path < num_paths; path++) {
        // Initialize local RNG state
        rng_state_t rng;
        init_rng(&rng, seeds[path]);
        
        double current_price = S0;
        
        // Simulate path with optimized loop
        #pragma acc loop seq
        for (int step = 0; step < steps; step++) {
            double z = rng_normal(&rng);
            double drift = drift_coeff;
            double diffusion = diffusion_coeff * z;
            current_price *= exp(drift + diffusion);
        }
        
        final_prices[path] = current_price;
    }
}

/**
 * Batched simulation for large datasets
 * Processes data in chunks to manage memory efficiently
 */
void monte_carlo_batched(double S0, double mu, double sigma, double T,
                        int steps, int total_paths, double **all_prices,
                        int batch_size) {
    
    double *batch_prices = (double*)malloc(batch_size * sizeof(double));
    unsigned long long *batch_seeds = (unsigned long long*)malloc(batch_size * sizeof(unsigned long long));
    
    // Initialize seeds for reproducible results
    srand(time(NULL));
    for (int i = 0; i < batch_size; i++) {
        batch_seeds[i] = rand();
    }
    
    #pragma acc data create(batch_prices[0:batch_size], batch_seeds[0:batch_size])
    {
        for (int batch_start = 0; batch_start < total_paths; batch_start += batch_size) {
            int current_batch_size = (batch_start + batch_size > total_paths) ? 
                                   (total_paths - batch_start) : batch_size;
            
            // Update seeds for this batch
            #pragma acc parallel loop present(batch_seeds[0:batch_size])
            for (int i = 0; i < current_batch_size; i++) {
                batch_seeds[i] = (batch_start + i) * 12345 + 67890;
            }
            
            // Run simulation for this batch
            monte_carlo_kernel_optimized(S0, mu, sigma, T, steps, current_batch_size,
                                       batch_prices, batch_seeds);
            
            // Copy results back
            #pragma acc update host(batch_prices[0:current_batch_size])
            
            // Store in final array
            for (int i = 0; i < current_batch_size; i++) {
                (*all_prices)[batch_start + i] = batch_prices[i];
            }
        }
    }
    
    free(batch_prices);
    free(batch_seeds);
}

/**
 * Multi-GPU implementation using async operations
 * Distributes work across multiple devices
 */
void monte_carlo_multi_gpu(double S0, double mu, double sigma, double T,
                          int steps, int num_paths, double *final_prices) {
    
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    if (num_devices <= 0) {
        printf("Warning: No NVIDIA devices found, using single device\n");
        num_devices = 1;
    }
    
    printf("Using %d GPU device(s)\n", num_devices);
    
    int paths_per_device = num_paths / num_devices;
    int remaining_paths = num_paths % num_devices;
    
    // Allocate device-specific arrays
    double **device_prices = (double**)malloc(num_devices * sizeof(double*));
    unsigned long long **device_seeds = (unsigned long long**)malloc(num_devices * sizeof(unsigned long long*));
    
    for (int dev = 0; dev < num_devices; dev++) {
        acc_set_device_num(dev, acc_device_nvidia);
        
        int device_paths = paths_per_device + ((dev < remaining_paths) ? 1 : 0);
        
        device_prices[dev] = (double*)malloc(device_paths * sizeof(double));
        device_seeds[dev] = (unsigned long long*)malloc(device_paths * sizeof(unsigned long long));
        
        // Initialize seeds for this device
        for (int i = 0; i < device_paths; i++) {
            device_seeds[dev][i] = (dev * paths_per_device + i) * 12345 + 67890;
        }
        
        // Create data on device asynchronously
        #pragma acc enter data create(device_prices[dev][0:device_paths], \
                                      device_seeds[dev][0:device_paths]) \
                              async(dev)
        
        #pragma acc update device(device_seeds[dev][0:device_paths]) async(dev)
        
        // Launch kernel asynchronously
        #pragma acc parallel loop present(device_prices[dev][0:device_paths], \
                                          device_seeds[dev][0:device_paths]) \
                                 async(dev)
        for (int path = 0; path < device_paths; path++) {
            rng_state_t rng;
            init_rng(&rng, device_seeds[dev][path]);
            
            double current_price = S0;
            double dt = T / steps;
            double drift_coeff = (mu - 0.5 * sigma * sigma) * dt;
            double diffusion_coeff = sigma * sqrt(dt);
            
            #pragma acc loop seq
            for (int step = 0; step < steps; step++) {
                double z = rng_normal(&rng);
                current_price *= exp(drift_coeff + diffusion_coeff * z);
            }
            
            device_prices[dev][path] = current_price;
        }
    }
    
    // Synchronize and collect results
    int result_offset = 0;
    for (int dev = 0; dev < num_devices; dev++) {
        int device_paths = paths_per_device + ((dev < remaining_paths) ? 1 : 0);
        
        // Wait for this device and copy results
        #pragma acc wait(dev)
        #pragma acc update host(device_prices[dev][0:device_paths]) async(dev)
        #pragma acc wait(dev)
        
        // Copy to final array
        memcpy(&final_prices[result_offset], device_prices[dev], 
               device_paths * sizeof(double));
        result_offset += device_paths;
        
        // Clean up device memory
        #pragma acc exit data delete(device_prices[dev][0:device_paths], \
                                     device_seeds[dev][0:device_paths])
        
        free(device_prices[dev]);
        free(device_seeds[dev]);
    }
    
    free(device_prices);
    free(device_seeds);
}

/**
 * Advanced statistics calculation with parallel reductions
 */
void calculate_advanced_statistics(double *prices, int num_paths,
                                  double *mean, double *variance, 
                                  double *skewness, double *kurtosis) {
    
    // First pass: calculate mean
    double sum = 0.0;
    #pragma acc parallel loop reduction(+:sum) present(prices[0:num_paths])
    for (int i = 0; i < num_paths; i++) {
        sum += prices[i];
    }
    *mean = sum / num_paths;
    
    // Second pass: calculate higher-order moments
    double sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    double mean_val = *mean;
    
    #pragma acc parallel loop reduction(+:sum2,sum3,sum4) \
                             present(prices[0:num_paths]) \
                             private(mean_val)
    for (int i = 0; i < num_paths; i++) {
        double diff = prices[i] - mean_val;
        double diff2 = diff * diff;
        double diff3 = diff2 * diff;
        double diff4 = diff2 * diff2;
        
        sum2 += diff2;
        sum3 += diff3;
        sum4 += diff4;
    }
    
    *variance = sum2 / num_paths;
    double std_dev = sqrt(*variance);
    *skewness = (sum3 / num_paths) / (std_dev * std_dev * std_dev);
    *kurtosis = (sum4 / num_paths) / (*variance * *variance) - 3.0;
}

int main() {
    printf("Advanced OpenACC Monte Carlo Stock Price Simulation\n");
    printf("==================================================\n");
    
    // Simulation parameters
    const double S0 = 100.0;
    const double mu = 0.08;
    const double sigma = 0.25;
    const double T = 1.0;
    const int steps = 252;
    const int num_paths = 500000;  // Larger simulation
    
    printf("Parameters:\n");
    printf("  Initial Price: $%.2f\n", S0);
    printf("  Drift: %.2f%%\n", mu * 100);
    printf("  Volatility: %.2f%%\n", sigma * 100);
    printf("  Time Horizon: %.1f year\n", T);
    printf("  Steps: %d\n", steps);
    printf("  Paths: %d\n\n", num_paths);
    
    // Allocate memory
    double *final_prices = (double*)malloc(num_paths * sizeof(double));
    
    // Run advanced multi-GPU simulation
    printf("Running multi-GPU OpenACC simulation...\n");
    clock_t start = clock();
    
    monte_carlo_multi_gpu(S0, mu, sigma, T, steps, num_paths, final_prices);
    
    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Calculate advanced statistics on GPU
    #pragma acc data copyin(final_prices[0:num_paths])
    {
        double mean, variance, skewness, kurtosis;
        calculate_advanced_statistics(final_prices, num_paths,
                                    &mean, &variance, &skewness, &kurtosis);
        
        // Display results
        printf("\nAdvanced Simulation Results:\n");
        printf("============================\n");
        printf("Execution Time: %.4f seconds\n", elapsed_time);
        printf("Throughput: %.0f paths/second\n", num_paths / elapsed_time);
        printf("\nStatistical Moments:\n");
        printf("  Mean: $%.2f\n", mean);
        printf("  Std Dev: $%.2f\n", sqrt(variance));
        printf("  Skewness: %.4f\n", skewness);
        printf("  Kurtosis: %.4f\n", kurtosis);
        
        // Theoretical comparison
        double theo_mean = S0 * exp(mu * T);
        double theo_var = S0 * S0 * exp(2 * mu * T) * (exp(sigma * sigma * T) - 1);
        
        printf("\nComparison with Theory:\n");
        printf("======================\n");
        printf("Theoretical Mean: $%.2f (Error: %.3f%%)\n", 
               theo_mean, 100.0 * fabs(mean - theo_mean) / theo_mean);
        printf("Theoretical Variance: %.2f (Error: %.3f%%)\n", 
               theo_var, 100.0 * fabs(variance - theo_var) / theo_var);
        
        // Risk metrics
        printf("\nRisk Metrics:\n");
        printf("=============\n");
        printf("VaR (5%%): $%.2f\n", mean - 1.645 * sqrt(variance));
        printf("CVaR (5%%): $%.2f\n", mean - 2.0 * sqrt(variance));
    }
    
    free(final_prices);
    
    printf("\nSimulation completed successfully!\n");
    return 0;
}