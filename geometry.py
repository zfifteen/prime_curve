"""
Planned Enhancements for geometry.py
-------------------------------------

1. **Domain Curvature Transformation**:
    - Refactor prime gap calculations by normalizing Δₙ against Δmax.
    - Focus computations on low-curvature regions to reduce search space.

2. **Universal Form Transformer**:
    - Introduce Lorentz-like boosts using velocity-ratio mappings (v/c).
    - Predict prime distributions using ratios like the golden ratio for Δmax.

3. **Fourier and GMM Integration**:
    - Implement Fourier fits and GMM clustering to identify prime density patterns.
    - Use these tools to refine curvature transformations and optimize efficiency.

4. **Dynamic Origin with Invariant C**:
    - Treat Δmax as an adaptive limit derived from primorial products or Euler's constant.
    - Enhance prime-finding efficiency by prioritizing zero-curvature paths.

5. **Extended Statistical Analysis**:
    - Add statistical summaries of prime distributions and growth factors.
    - Highlight clustering patterns and anomalies in prime sequences.

6. **Multi-Domain Applications**:
    - Extend the code to explore applications in physical simulations.
    - Use curvature insights to model particle velocities or turbulent flows.
"""

# ------------------------------------------------------------------------------
# 1. Constants and primes
# ------------------------------------------------------------------------------
import numpy as np
from sklearn.mixture import GaussianMixture
from sympy import sieve
import warnings

warnings.filterwarnings("ignore")

phi = (1 + np.sqrt(5)) / 2
N_MAX = 20000
primes_list = list(sieve.primerange(2, N_MAX + 1))

# ------------------------------------------------------------------------------
# 2. Domain Curvature Transformation
# ------------------------------------------------------------------------------
def compute_prime_gaps(primes):
    """
    Compute normalized prime gaps Δₙ/Δmax.
    """
    gaps = np.diff(primes)
    delta_max = np.log(primes[-1])**2  # Approximation from prime number theorem
    return gaps / delta_max

normalized_gaps = compute_prime_gaps(primes_list)

# ------------------------------------------------------------------------------
# 3. Fourier and GMM Integration for Density Patterns
# ------------------------------------------------------------------------------
def fit_density_models(data, nbins=20, components=5):
    """
    Fit both Fourier series and GMM to the data.
    """
    # Fourier Fit
    x = (data % phi) / phi
    y, edges = np.histogram(data, bins=nbins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2 / phi

    def design_matrix(x, M=5):
        cols = [np.ones_like(x)]
        for k in range(1, M + 1):
            cols.append(np.cos(2 * np.pi * k * x))
            cols.append(np.sin(2 * np.pi * k * x))
        return np.vstack(cols).T

    A = design_matrix(centers)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    fourier_amplitudes = coeffs[1:]

    # GMM Fit
    X = x.reshape(-1, 1)
    gm = GaussianMixture(n_components=components, random_state=0).fit(X)
    gmm_means = gm.means_.flatten()
    gmm_sigmas = np.sqrt(gm.covariances_.flatten())

    return fourier_amplitudes, gmm_means, gmm_sigmas

# ------------------------------------------------------------------------------
# 4. Dynamic Origin with Invariant C
# ------------------------------------------------------------------------------
def compute_dynamic_origin(primes):
    """
    Use primorial products or Euler's constant to adapt Δmax dynamically.
    """
    primorial = np.prod(primes[:5])  # First 5 primes
    dynamic_origin = primorial / np.euler_gamma
    return dynamic_origin

dynamic_c = compute_dynamic_origin(primes_list)

# ------------------------------------------------------------------------------
# 5. Statistical Analysis and Visualization
# ------------------------------------------------------------------------------
def statistical_summary(primes, gaps):
    """
    Compute and print statistical summaries of the prime distribution.
    """
    print("\n=== Statistical Summary ===")
    print(f"Total Primes: {len(primes)}")
    print(f"Mean Gap: {np.mean(gaps):.2f}")
    print(f"Median Gap: {np.median(gaps):.2f}")
    print(f"Max Gap: {np.max(gaps):.2f}")
    print(f"Min Gap: {np.min(gaps):.2f}")

    # Prime distribution stats
    prime_array = np.array(primes)
    print("\nPrime Distribution Statistics:")
    print(f"Mean of Primes: {np.mean(prime_array):.2f}")
    print(f"Median of Primes: {np.median(prime_array):.2f}")
    print(f"Standard Deviation: {np.std(prime_array):.2f}")

statistical_summary(primes_list, normalized_gaps)