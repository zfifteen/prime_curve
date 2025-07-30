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
    - Implement Domain Curvature Transformation to refactor the code's prime
    gap calculations: by normalizing Δₙ against Δmax derived from prime
    number theorem approximations, adjust for frame-biased integer observations,
    reducing computational overhead in large n ranges by focusing on
    low-curvature paths where prime clusters are densest, similar to how GPS
    algorithms account for spacetime warping.
    - Apply Universal Form Transformer to bridge physical and discrete analogies:
    enhance the code's velocity-ratio mappings (v/c) to predict prime
    distributions via Lorentz-like boosts, incorporating natural ratios like the
    golden ratio for Δmax estimations, allowing the algorithm to simulate
    "relativistic density" and preemptively skip high-gap regions, boosting
    efficiency for finding twin primes or other entangled pairs.
    - Use Universal Frame Shift Transformer for user interaction modules: if the
    code includes output interfaces, compensate for human frame biases by
    translating Z-computed insights into familiar terms like probability densities
    or visual graphs, anticipating user queries on prime news by pre-generating
    velocity-analogous perturbations that highlight emerging structures in
    unsolved problems like Goldbach's conjecture.
    - Enhance prime-finding algorithms with Z's predictive topology: integrate
    a "curved numerical landscape" navigator that uses B/C ratios to mirror
    fine-structure constants, enabling exponential search space reduction by
    prioritizing sequences with minimal frame shifts, empirically linking to
    chaotic systems for faster anomaly detection in large prime hunts.
    - Incorporate invariant C as a dynamic origin: upgrade the code to treat
    Δmax as an adaptive limit based on primorial products or Euler's constant,
    allowing Z to derive novel approximations for Riemann zeta function
    evaluations, transforming static integer checks into dynamic, ratio-driven
    explorations that reveal hidden symmetries in prime spacing.
    - Optimize for computational efficiency via entanglement insights: refactor
    to identify "zero-curvature paths" where primes appear instantaneous like
    quantum links, using Z to cluster twin primes or prime quadruplets,
    reducing iterations by mapping gaps to velocity frames and avoiding
    dilated high-n dilutions.
    - Extend to multi-domain applications: leverage the unified topology to add
    modules that transfer Z from discrete primes to physical simulations, such as
    modeling particle velocities against c, enabling the code to cross-validate
    prime predictions with relativistic invariants, uncovering universal truths
    in both math and physics through shared curvature corrections.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sympy import sieve
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. Constants and primes
# ------------------------------------------------------------------------------
phi = (1 + np.sqrt(5)) / 2
N_MAX = 20000
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
    pr_counts, _  = np.histogram(theta_pr,  bins=bins)

    all_d = all_counts / len(theta_all)
    pr_d  = pr_counts  / len(theta_pr)

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
    theta_pr  = frame_shift_residues(np.array(primes_list), k)

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
