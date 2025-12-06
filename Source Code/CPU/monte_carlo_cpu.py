"""
CPU Baseline Implementation for Monte Carlo Stock Price Simulation

This module provides a sequential CPU implementation of Monte Carlo simulation
for financial derivative pricing using geometric Brownian motion.

Course: 251-COE-506-01 (GPU Programming & Architecture)
Institution: King Fahd University of Petroleum & Minerals
"""

import numpy as np


def monte_carlo_cpu(S0, mu, sigma, T, steps, num_paths):
    """
    Sequential CPU implementation of Monte Carlo stock price simulation.

    Implements the Euler-Maruyama discretization of geometric Brownian motion:
    S(t+Δt) = S(t) × exp[(μ - σ²/2)Δt + σ√Δt × Z]

    Parameters
    ----------
    S0 : float
        Initial stock price
    mu : float
        Drift coefficient (expected return)
    sigma : float
        Volatility coefficient (standard deviation of returns)
    T : float
        Time horizon in years
    steps : int
        Number of time discretization steps
    num_paths : int
        Number of independent Monte Carlo simulation paths

    Returns
    -------
    prices : ndarray, shape (num_paths, steps+1)
        Simulated stock price trajectories
        prices[i, j] represents price of path i at time step j

    Complexity
    ----------
    Time: O(num_paths × steps)
    Space: O(num_paths × steps)

    Notes
    -----
    This is the baseline reference implementation. It is intentionally
    non-optimized to clearly demonstrate the sequential bottleneck that
    GPU acceleration addresses.
    """
    dt = T / steps
    prices = np.zeros((num_paths, steps + 1), dtype=np.float64)
    prices[:, 0] = S0

    # ========================================================================
    # CRITICAL SECTION: Sequential nested loops
    # This is the primary computational bottleneck on CPU
    # Total iterations: num_paths × steps = O(10^6 × 10^2) = O(10^8)
    # ========================================================================
    for path in range(num_paths):
        for step in range(1, steps + 1):
            # Random number generation (expensive operation)
            z = np.random.standard_normal()

            # Compute price evolution components
            drift = (mu - 0.5 * sigma**2) * dt        # O(1) arithmetic ops
            diffusion = sigma * np.sqrt(dt) * z       # O(1) + sqrt overhead

            # Update price using exponential formula (transcendental function)
            prices[path, step] = prices[path, step-1] * np.exp(drift + diffusion)

    return prices


if __name__ == "__main__":
    # Example usage
    import time

    # Simulation parameters
    S0 = 100.0      # Initial stock price
    mu = 0.08       # Drift (8% annual)
    sigma = 0.25    # Volatility (25% annual)
    T = 1.0         # Time horizon (1 year)
    steps = 252     # Time steps (trading days)
    num_paths = 10000  # Number of simulation paths

    print("Running CPU Monte Carlo Simulation...")
    print(f"Paths: {num_paths:,}, Steps: {steps}")

    start_time = time.time()
    prices = monte_carlo_cpu(S0, mu, sigma, T, steps, num_paths)
    elapsed_time = time.time() - start_time

    # Calculate statistics
    final_prices = prices[:, -1]
    mean_price = np.mean(final_prices)
    std_price = np.std(final_prices)

    print(f"\nExecution Time: {elapsed_time:.4f} seconds")
    print(f"Mean Final Price: ${mean_price:.2f}")
    print(f"Std Dev Final Price: ${std_price:.2f}")
    print(f"Throughput: {num_paths/elapsed_time:,.0f} paths/second")


