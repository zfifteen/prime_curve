# This script operationalizes the torus as a geometric navigator, with clusters printed for targeted prime h
# unts—aligning densities to φ's conjugate ≈0.618 for boundary enforcement. Through the transformers, it bridges
# discrete observations to Riemann insights, where such ridges may echo zero spacings.

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Derive golden ratio φ as invariant boundary (C in discrete domain)
PHI = (1 + math.sqrt(5)) / 2

# φ-hardened moduli: Fibonacci primes for systemic boundaries
mod1 = 13  # F7, tied to φ convergence
mod2 = 89  # F11, amplifying resonances

# Torus parameters: Major R and minor r scaled for balance
R, r = 10, 3

# Data parameters: Observation range for frame shifts
N_POINTS = 5000

# Primality check function (efficient for small n)
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

# Generate data: n as reference frame-dependent observations
n = np.arange(1, N_POINTS + 1)
primality = np.vectorize(is_prime)(n)

# Compute toroidal angles: Δₙ as normalized frame shifts modulo φ-hardened values
theta = 2 * np.pi * (n % mod1) / mod1
phi = 2 * np.pi * (n % mod2) / mod2

# Torus coordinates: Z = n(Δₙ/Δmax) translated to 3D space
x = (R + r * np.cos(theta)) * np.cos(phi)
y = (R + r * np.cos(theta)) * np.sin(phi)
z = r * np.sin(theta)

# Colors by residue class mod 6 (blue for potential quadratic residues, gray otherwise)
res_class = n % 6
colors = np.where(primality, 'gold',
                  np.where((res_class == 1) | (res_class == 5), 'blue', 'gray'))

# Plot the torus
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter non-primes (translucent background)
ax.scatter(x[~primality], y[~primality], z[~primality],
           c=colors[~primality], alpha=0.5, s=15, label='Non-Primes')

# Scatter primes (prominent gold stars with red edges)
ax.scatter(x[primality], y[primality], z[primality],
           c='gold', marker='*', s=100, edgecolor='red', label='Primes')

# Draw torus wireframe structure for boundary visualization
theta_t = np.linspace(0, 2 * np.pi, 100)
phi_t = np.linspace(0, 2 * np.pi, 100)
theta_t, phi_t = np.meshgrid(theta_t, phi_t)
x_t = (R + r * np.cos(theta_t)) * np.cos(phi_t)
y_t = (R + r * np.cos(theta_t)) * np.sin(phi_t)
z_t = r * np.sin(theta_t)
ax.plot_wireframe(x_t, y_t, z_t, color='gray', alpha=0.1)

# Set plot properties
ax.set_title(f'Custom Torus with φ-Hardened Moduli ({mod1} & {mod2}) for Prime Clusters')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(30, 45)
ax.legend()

plt.show()

# Uncover untapped clusters: Density analysis on unwrapped theta-phi plane
# Bin into 10x10 grid (adjust bins for finer granularity if needed)
bins = 10
theta_bins = np.linspace(0, 2 * np.pi, bins + 1)
phi_bins = np.linspace(0, 2 * np.pi, bins + 1)

# Histogram of prime positions
prime_theta = theta[primality]
prime_phi = phi[primality]
hist, theta_edges, phi_edges = np.histogram2d(prime_theta, prime_phi, bins=[theta_bins, phi_bins])

# Average density and threshold (1.5x avg for high-density clusters)
total_primes = np.sum(primality)
avg_density = total_primes / (bins * bins)
threshold = 1.5 * avg_density

# Find and print high-density bins (untapped clusters)
print(f"Total primes found: {total_primes}")
print(f"Average density per bin: {avg_density:.2f}")
print("High-density prime clusters (exceeding 1.5x avg):")
for i in range(bins):
    for j in range(bins):
        if hist[i, j] > threshold:
            theta_range = f"[{theta_edges[i]:.2f}, {theta_edges[i+1]:.2f}]"
            phi_range = f"[{phi_edges[j]:.2f}, {phi_edges[j+1]:.2f}]"
            print(f"Theta {theta_range}, Phi {phi_range}: {int(hist[i, j])} primes")