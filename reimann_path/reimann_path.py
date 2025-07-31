import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpmath

mpmath.mp.dps = 15  # Reduced for speed

# Efficient prime generation using Sieve of Eratosthenes
def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [n for n in range(2, limit + 1) if sieve[n]]

# Constants
phi = (1 + np.sqrt(5)) / 2
k = 0.300

# Generate first 10000 primes
primes = generate_primes(110000)[:100]

# Compute gaps and max gap
gaps = np.diff(primes)
max_gap = max(gaps)

# Optimized Z_list
Z_list = [primes[i] * (gaps[i] / max_gap) ** k for i in range(len(gaps))]

# Compute first 10000 non-trivial zeros' imaginary parts
known_zeros_im = [float(mpmath.im(mpmath.zetazero(n))) for n in range(1, 10001)]

# Set up coarser grid for surface
max_imag = max(Z_list) * 1.1 if Z_list else 100
real = np.linspace(0.1, 1.2, 20)  # Coarser
imag = np.linspace(0, max_imag, 50)  # Coarser
Re, Im = np.meshgrid(real, imag)

# Compute log|zeta| on grid
zeta_mag = np.zeros(Re.shape)
for i in range(Re.shape[0]):
    for j in range(Re.shape[1]):
        s = complex(Re[i, j], Im[i, j])
        zeta = mpmath.zeta(s)
        abs_zeta = abs(zeta)
        zeta_mag[i, j] = float(mpmath.log(abs_zeta)) if abs_zeta > 0 else -10

# Plot the surface
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Re, Im, zeta_mag, cmap='viridis', alpha=0.7)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='log|ζ(s)|')

# Overlay frame-shifted primes (subsample every 10th for clarity)
for idx, z_im in enumerate(Z_list[::10]):
    s = complex(0.5, z_im)
    zeta = mpmath.zeta(s)
    abs_zeta = abs(zeta)
    log_abs = float(mpmath.log(abs_zeta)) if abs_zeta > 0 else -10
    adjusted_log = log_abs * (z_im / phi)
    ax.scatter(0.5, z_im, adjusted_log, c='red', s=50, edgecolor='gold', label='Shifted Primes' if idx == 0 else None)

# Overlay zeros (subsample every 10th)
min_log = np.min(zeta_mag) - 5
for idx, t_zero in enumerate(known_zeros_im[::10]):
    if t_zero <= max_imag:
        ax.scatter(0.5, t_zero, min_log, c='blue', s=100, marker='x', label='Zero Approx' if idx == 0 else None)

ax.set_title('Optimized Riemann Zeta Landscape (n=10000): Curvature-Adjusted Frame-Shifted Primes and Extended Zero Approximations')
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part (Curvature-Adjusted for Primes)')
ax.set_zlabel('log|ζ(s)| (Ratio-Scaled)')
ax.legend()
ax.view_init(elev=30, azim=135)
plt.show()