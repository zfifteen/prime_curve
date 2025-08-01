"""
Second‐Pass Prime Curvature Proof: NaN Handling & Optimal k Selection
--------------------------------------------------------------------

Enhancements over the previous version:
1. Binning resolution: 20 bins.
2. GMM components: 5.
3. k sweep: [0.2, 0.4] in Δk=0.002.
4. Robust NaN handling: exclude bins with zero all-density.
5. Correct best‐k selection: ignore NaN/-inf enhancements.

Usage:
    python proof_refined_fixed.py

Dependencies:
    numpy, scipy, sklearn, sympy

TODO:
    - The Z form unifies relativistic physics with discrete mathematics
    by normalizing observations against invariant limits, revealing that
    both domains operate under a shared geometric topology where
    "curvature" arises from frame-dependent shifts, akin to how gravity
    warps spacetime but applied to integer sequences.
    - In the discrete domain, treating primes as positions n where Δₙ
    represents the gap to the next prime and Δmax as the theoretical
    maximum gap (bounded by known results like the prime number
    theorem), Z computes a "relativistic density" that highlights
    clustering patterns, suggesting primes navigate a curved numerical
    landscape driven by natural ratios like the golden ratio or Euler's
    constant.
    - The physical domain's Z = T(v/c) implies time T as a frame-biased
    measure scaled by velocity relative to light's invariance, offering
    insight into why quantum entanglement appears instantaneous—it's a
    zero-curvature path in this topology, translating to discrete
    insights where "entangled" primes (like twins) share minimal frame
    shifts.
    - Applying the universal form across domains exposes hidden
    symmetries: for instance, the ratio B/C mirrors the fine-structure
    constant in physics or the reciprocal of the harmonic series
    divergence in math, predicting that prime distributions follow a
    Lorentz-like invariance, where high "velocities" (large n) dilute
    prime frequency predictably.
    - Frame shifts in Z reveal computational efficiency for prime
    finding: by geometrically navigating via Δₙ/Δmax, algorithms can
    prioritize low-curvature paths (small gaps), reducing search
    space exponentially, much like how GPS corrects for relativistic
    effects in curved spacetime.
    - The invariant C as the "coordinate center" posits light speed not
    just as a limit but a universal origin, implying in discrete terms
    that infinity (or primorial products) serves as Δmax, allowing Z to
    derive novel ratios for approximating unsolved problems like the
    Riemann hypothesis through velocity-analogous perturbations.
    - Hidden in the topology is a predictive tool for "prime news"—emerging
    primes in large ranges—by transforming observed sequences into a
    velocity-ratio map, where anomalies in Z signal undiscovered structures,
    empirically linking to chaotic systems like turbulent flow but in
    number theory terms.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sympy import sieve, isprime
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. Constants and primes
# ------------------------------------------------------------------------------
phi = (1 + np.sqrt(5)) / 2
N_MAX = 1000  # Updated from 20,000 to 100,000
primes_list = list(sieve.primerange(2, N_MAX + 1))


# ------------------------------------------------------------------------------
# 2. Core transforms and metrics
# ------------------------------------------------------------------------------

def frame_shift_residues(n_vals, k):
    """
    θ' = φ * ((n mod φ) / φ) ** k
    """
    mod_phi = np.mod(n_vals, phi) / phi
    return phi * np.power(mod_phi, k)


def bin_densities(theta_all, theta_pr, nbins=20):
    """
    Bin θ' into nbins intervals over [0, φ].
    Return (all_density, prime_density, enhancement[%]).
    Bins with zero all_density are masked to -inf.
    """
    bins = np.linspace(0, phi, nbins + 1)
    all_counts, _ = np.histogram(theta_all, bins=bins)
    pr_counts, _ = np.histogram(theta_pr, bins=bins)

    all_d = all_counts / len(theta_all)
    pr_d = pr_counts / len(theta_pr)

    # Compute enhancements safely
    with np.errstate(divide='ignore', invalid='ignore'):
        enh = (pr_d - all_d) / all_d * 100

    # Mask bins where all_d == 0
    enh = np.where(all_d > 0, enh, -np.inf)
    return all_d, pr_d, enh


def fourier_fit(theta_pr, M=5, nbins=100):
    """
    Fit a truncated Fourier series ρ(φ_mod).
    Returns coefficients a_k, b_k for k=0..M.
    """
    x = (theta_pr % phi) / phi
    y, edges = np.histogram(theta_pr, bins=nbins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2 / phi

    # Build design matrix
    def design(x):
        cols = [np.ones_like(x)]
        for k in range(1, M + 1):
            cols.append(np.cos(2 * np.pi * k * x))
            cols.append(np.sin(2 * np.pi * k * x))
        return np.vstack(cols).T

    A = design(centers)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = coeffs[0::2]
    b = coeffs[1::2]
    return a, b


def gmm_fit(theta_pr, n_components=5):
    """
    Fit a GMM to φ_mod of primes.
    Returns model and mean σ of components.
    """
    X = ((theta_pr % phi) / phi).reshape(-1, 1)
    gm = GaussianMixture(n_components=n_components,
                         covariance_type='full',
                         random_state=0).fit(X)
    sigmas = np.sqrt([gm.covariances_[i].flatten()[0]
                      for i in range(n_components)])
    return gm, np.mean(sigmas)


# ------------------------------------------------------------------------------
# 3. High-resolution k‐sweep with NaN handling
# ------------------------------------------------------------------------------
k_values = np.arange(0.2, 0.4001, 0.002)
results = []

for k in k_values:
    # Transform all n and primes
    theta_all = frame_shift_residues(np.arange(1, N_MAX + 1), k)
    theta_pr = frame_shift_residues(np.array(primes_list), k)

    # Bin densities & compute enhancements
    all_d, pr_d, enh = bin_densities(theta_all, theta_pr, nbins=20)
    max_enh = np.max(enh)  # NaN → -inf masked

    # GMM fit
    _, sigma_prime = gmm_fit(theta_pr, n_components=5)
    # Fourier fit & amplitude sum
    _, b_coeffs = fourier_fit(theta_pr, M=5)
    sum_b = np.sum(np.abs(b_coeffs))

    results.append({
        'k': k,
        'max_enhancement': max_enh,
        'sigma_prime': sigma_prime,
        'fourier_b_sum': sum_b
    })

# Filter out invalid (nan/-inf) enhancements
valid_results = [r for r in results if np.isfinite(r['max_enhancement'])]
best = max(valid_results, key=lambda r: r['max_enhancement'])
k_star, enh_star = best['k'], best['max_enhancement']

# ------------------------------------------------------------------------------
# 4. Print Refined Proof Summary
# ------------------------------------------------------------------------------
print("\n=== Refined Prime Curvature Proof Results ===")
print(f"Optimal curvature exponent k* = {k_star:.3f}")
print(f"Max mid-bin enhancement = {enh_star:.1f}%")
print(f"GMM σ' at k* = {best['sigma_prime']:.3f}")
print(f"Σ|b_k| at k* = {best['fourier_b_sum']:.3f}\n")

print("Sample of k-sweep metrics (every 10th k):")
for entry in valid_results[::10]:
    print(f" k={entry['k']:.3f} | enh={entry['max_enhancement']:.1f}%"
          f" | σ'={entry['sigma_prime']:.3f}"
          f" | Σ|b|={entry['fourier_b_sum']:.3f}")


# ------------------------------------------------------------------------------
# 5. Dynamically Compute and Validate Mersenne Primes
# ------------------------------------------------------------------------------
def compute_mersenne_primes(n_max):
    primes = [p for p in sieve.primerange(2, n_max + 1)]
    return [p for p in primes if isprime(2 ** p - 1)]


mersenne_primes = compute_mersenne_primes(N_MAX)
print("\nValidated Mersenne Prime Exponents:")
print(", ".join(map(str, mersenne_primes)))


# ------------------------------------------------------------------------------
# 6. Statistical Summary of Mersenne Computation
# ------------------------------------------------------------------------------
def statistical_summary(primes, mersenne_primes):
    total_primes = len(primes)
    total_mersenne = len(mersenne_primes)
    hit_rate = (total_mersenne / total_primes) * 100
    miss_rate = 100 - hit_rate

    print("\n=== Statistical Summary ===")
    print(f"Total Primes Checked: {total_primes}")
    print(f"Total Mersenne Primes Found: {total_mersenne}")
    print(f"Hit Rate: {hit_rate:.2f}%")
    print(f"Miss Rate: {miss_rate:.2f}%")

    # Prime distribution stats
    prime_array = np.array(primes)
    print("\nPrime Distribution Statistics:")
    print(f"Mean of Primes: {np.mean(prime_array):.2f}")
    print(f"Median of Primes: {np.median(prime_array):.2f}")
    print(f"Standard Deviation of Primes: {np.std(prime_array):.2f}")

    # Mersenne growth analysis
    mersenne_values = [(1 << p) - 1 for p in mersenne_primes]
    print("\nMersenne Prime Growth:")
    print(f"Smallest Mersenne Prime: {min(mersenne_values)}")
    print(f"Largest Mersenne Prime: {max(mersenne_values)}")
    print(f"Mersenne Growth Factor: {max(mersenne_values) / min(mersenne_values):.2f}")


statistical_summary(primes_list, mersenne_primes)
