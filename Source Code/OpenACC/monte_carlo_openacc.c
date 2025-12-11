/*
 * OpenACC Implementation for Monte Carlo Stock Price Simulation
 * 
 * This module provides an OpenACC accelerated implementation of Monte Carlo simulation
 * for financial derivative pricing using geometric Brownian motion.
 * 
 * Course: 251-COE-506-01 (GPU Programming & Architecture)
 * Institution: King Fahd University of Petroleum & Minerals
 * 
 * Compilation:
 *   pgcc -acc -Minfo=accel -fast monte_carlo_openacc.c -lm -o monte_carlo_openacc
 * 
 * Alternative compilers:
 *   gcc -fopenacc -O3 monte_carlo_openacc.c -lm -o monte_carlo_openacc
 *   nvc -acc -Minfo=accel -fast monte_carlo_openacc.c -lm -o monte_carlo_openacc
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Box-Muller transform to generate normally distributed random numbers
 * Uses uniform random numbers to generate standard normal variates
 */
void box_muller_transform(double u1, double u2, double *z1, double *z2) {
    double magnitude = sqrt(-2.0 * log(u1));
    double angle = 2.0 * M_PI * u2;
    *z1 = magnitude * cos(angle);
    *z2 = magnitude * sin(angle);
}

/**
 * OpenACC accelerated Monte Carlo stock price simulation
 * 
 * Implements the Euler-Maruyama discretization of geometric Brownian motion:
 * S(t+Δt) = S(t) × exp[(μ - σ²/2)Δt + σ√Δt × Z]
 * 
 * Parameters:
 *   S0: Initial stock price
 *   mu: Drift coefficient (expected return)
 *   sigma: Volatility coefficient (standard deviation of returns)
 *   T: Time horizon in years
 *   steps: Number of time discretization steps
 *   num_paths: Number of independent Monte Carlo simulation paths
 *   prices: Output array [num_paths][steps+1] for price trajectories
 */
void monte_carlo_openacc(double S0, double mu, double sigma, double T, 
                        int steps, int num_paths, double **prices) {
    
    double dt = T / steps;
    double drift_coeff = (mu - 0.5 * sigma * sigma) * dt;
    double diffusion_coeff = sigma * sqrt(dt);
    
    // Initialize first column with S0
    #pragma acc parallel loop present(prices[0:num_paths][0:steps+1])
    for (int path = 0; path < num_paths; path++) {
        prices[path][0] = S0;
    }
    
    // Prepare random numbers on host (OpenACC doesn't have good built-in RNG)
    double *random_numbers = (double*)malloc(num_paths * steps * sizeof(double));
    srand(time(NULL));
    
    // Generate random numbers on host
    for (int i = 0; i < num_paths * steps; i++) {
        // Generate uniform [0,1) random numbers
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        
        // Use Box-Muller to get normal random number
        double z1, z2;
        box_muller_transform(u1, u2, &z1, &z2);
        random_numbers[i] = z1;
        
        // Use z2 for next iteration if available
        if (i + 1 < num_paths * steps) {
            random_numbers[++i] = z2;
        }
    }
    
    // Copy random numbers to device
    #pragma acc data copyin(random_numbers[0:num_paths*steps]) \
                     present(prices[0:num_paths][0:steps+1])
    {
        // Main simulation loop with OpenACC parallelization
        #pragma acc parallel loop collapse(2) \
                    private(drift_coeff, diffusion_coeff)
        for (int path = 0; path < num_paths; path++) {
            for (int step = 1; step <= steps; step++) {
                // Get random number for this path/step
                double z = random_numbers[path * steps + step - 1];
                
                // Calculate price evolution
                double drift = drift_coeff;
                double diffusion = diffusion_coeff * z;
                
                // Update price using exponential formula
                prices[path][step] = prices[path][step-1] * exp(drift + diffusion);
            }
        }
    }
    
    free(random_numbers);
}

/**
 * Alternative implementation with better memory access patterns
 * Processes time steps in the outer loop for better cache locality
 */
void monte_carlo_openacc_optimized(double S0, double mu, double sigma, double T, 
                                  int steps, int num_paths, double **prices) {
    
    double dt = T / steps;
    double drift_coeff = (mu - 0.5 * sigma * sigma) * dt;
    double diffusion_coeff = sigma * sqrt(dt);
    
    // Initialize first column with S0
    #pragma acc parallel loop present(prices[0:num_paths][0:steps+1])
    for (int path = 0; path < num_paths; path++) {
        prices[path][0] = S0;
    }
    
    // Generate all random numbers at once
    double *random_numbers = (double*)malloc(num_paths * steps * sizeof(double));
    srand(time(NULL));
    
    for (int i = 0; i < num_paths * steps; i++) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        
        double z1, z2;
        box_muller_transform(u1, u2, &z1, &z2);
        random_numbers[i] = z1;
        
        if (i + 1 < num_paths * steps) {
            random_numbers[++i] = z2;
        }
    }
    
    // Copy data to device and run simulation
    #pragma acc data copyin(random_numbers[0:num_paths*steps]) \
                     present(prices[0:num_paths][0:steps+1])
    {
        // Process one time step at a time for better memory locality
        for (int step = 1; step <= steps; step++) {
            #pragma acc parallel loop \
                        private(drift_coeff, diffusion_coeff)
            for (int path = 0; path < num_paths; path++) {
                double z = random_numbers[path * steps + step - 1];
                double drift = drift_coeff;
                double diffusion = diffusion_coeff * z;
                
                prices[path][step] = prices[path][step-1] * exp(drift + diffusion);
            }
        }
    }
    
    free(random_numbers);
}

/**
 * Allocate 2D array with contiguous memory layout
 * This ensures better memory access patterns on GPU
 */
double** allocate_2d_array(int rows, int cols) {
    double **array = (double**)malloc(rows * sizeof(double*));
    double *data = (double*)malloc(rows * cols * sizeof(double));
    
    for (int i = 0; i < rows; i++) {
        array[i] = data + i * cols;
    }
    
    return array;
}

/**
 * Free 2D array allocated with allocate_2d_array
 */
void free_2d_array(double **array) {
    free(array[0]);  // Free the data block
    free(array);     // Free the pointer array
}

/**
 * Calculate basic statistics of final prices
 */
void calculate_statistics(double **prices, int num_paths, int steps, 
                         double *mean, double *std_dev) {
    double sum = 0.0;
    double sum_sq = 0.0;
    
    #pragma acc parallel loop reduction(+:sum,sum_sq) \
                            present(prices[0:num_paths][0:steps+1])
    for (int path = 0; path < num_paths; path++) {
        double final_price = prices[path][steps];
        sum += final_price;
        sum_sq += final_price * final_price;
    }
    
    *mean = sum / num_paths;
    *std_dev = sqrt((sum_sq / num_paths) - (*mean) * (*mean));
}

int main() {
    // Simulation parameters
    const double S0 = 100.0;        // Initial stock price
    const double mu = 0.08;         // Drift (8% annual)
    const double sigma = 0.25;      // Volatility (25% annual)
    const double T = 1.0;           // Time horizon (1 year)
    const int steps = 252;          // Time steps (trading days)
    const int num_paths = 100000;   // Number of simulation paths
    
    printf("OpenACC Monte Carlo Stock Price Simulation\n");
    printf("==========================================\n");
    printf("Initial Price: $%.2f\n", S0);
    printf("Drift (mu): %.2f%%\n", mu * 100);
    printf("Volatility (sigma): %.2f%%\n", sigma * 100);
    printf("Time Horizon: %.1f year(s)\n", T);
    printf("Time Steps: %d\n", steps);
    printf("Simulation Paths: %d\n\n", num_paths);
    
    // Allocate memory for price trajectories
    double **prices = allocate_2d_array(num_paths, steps + 1);
    
    // Copy data to device
    #pragma acc data create(prices[0:num_paths][0:steps+1])
    {
        printf("Running OpenACC simulation...\n");
        
        // Run simulation
        clock_t start = clock();
        monte_carlo_openacc_optimized(S0, mu, sigma, T, steps, num_paths, prices);
        clock_t end = clock();
        
        // Calculate statistics on device
        double mean_price, std_price;
        calculate_statistics(prices, num_paths, steps, &mean_price, &std_price);
        
        // Copy final results back to host
        #pragma acc update host(prices[0:num_paths][steps:1])
        
        double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Display results
        printf("\nSimulation Results:\n");
        printf("==================\n");
        printf("Execution Time: %.4f seconds\n", elapsed_time);
        printf("Mean Final Price: $%.2f\n", mean_price);
        printf("Std Dev Final Price: $%.2f\n", std_price);
        printf("Throughput: %.0f paths/second\n", num_paths / elapsed_time);
        
        // Theoretical values for comparison
        double theoretical_mean = S0 * exp(mu * T);
        double theoretical_std = S0 * exp(mu * T) * sqrt(exp(sigma * sigma * T) - 1);
        
        printf("\nComparison with Theory:\n");
        printf("======================\n");
        printf("Theoretical Mean: $%.2f (Error: %.2f%%)\n", 
               theoretical_mean, 
               100.0 * fabs(mean_price - theoretical_mean) / theoretical_mean);
        printf("Theoretical Std: $%.2f (Error: %.2f%%)\n", 
               theoretical_std,
               100.0 * fabs(std_price - theoretical_std) / theoretical_std);
    }
    
    // Clean up
    free_2d_array(prices);
    
    return 0;
}