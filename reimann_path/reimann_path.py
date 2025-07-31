import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpmath

mpmath.mp.dps = 10  # Set precision for mpmath

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Generate first 1000 primes
N_MAX = 10000
primes = [n for n in range(2, N_MAX) if is_prime(n)][:1000]

# Compute gaps and max gap for frame shifts
gaps = np.diff(primes)
max_gap = max(gaps)
# Z_list for frame shifts: Z = p * (gap / max_gap) for first 999 primes
Z_list = [primes[i] * (gaps[i] / max_gap) for i in range(len(gaps))]

# Known imaginary parts of first 20 non-trivial zeros (approximated)
known_zeros_im = [
    14.135, 21.022, 25.011, 30.425, 32.935, 37.586, 40.919, 43.327,
    48.005, 49.774, 52.970, 56.446, 59.347, 60.832, 65.113, 67.080,
    69.546, 72.067, 75.705, 77.145
]

# Set up grid for surface, scaled to cover shifted Z range
max_imag = max(Z_list) * 1.1 if Z_list else 100
real = np.linspace(0.1, 1.2, 50)
imag = np.linspace(0, max_imag, 100)
Re, Im = np.meshgrid(real, imag)

# Compute log|zeta| on grid
zeta_mag = np.zeros(Re.shape)
for i in range(Re.shape[0]):
    for j in range(Re.shape[1]):
        s = complex(Re[i, j], Im[i, j])
        zeta = mpmath.zeta(s)
        abs_zeta = abs(zeta)
        zeta_mag[i, j] = float(mpmath.log(abs_zeta)) if abs_zeta > 0 else -10  # Cap -inf

# Plot the surface
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Re, Im, zeta_mag, cmap='viridis', alpha=0.7)

# Overlay frame-shifted primes on critical line
for z_im in Z_list:
    s = complex(0.5, z_im)
    zeta = mpmath.zeta(s)
    abs_zeta = abs(zeta)
    log_abs = float(mpmath.log(abs_zeta)) if abs_zeta > 0 else -10
    # Apply Universal Form Transformer adjustment: scale log_abs by (z_im / math.e)
    adjusted_log = log_abs * (z_im / math.e)
    ax.scatter(0.5, z_im, adjusted_log, c='red', s=50, edgecolor='gold', label='Shifted Primes' if z_im == Z_list[0] else None)

# Overlay approximate non-trivial zeros at capped low value
min_log = np.min(zeta_mag) - 5  # Below surface minima
for t_zero in known_zeros_im:
    if t_zero <= max_imag:
        ax.scatter(0.5, t_zero, min_log, c='blue', s=100, marker='x', label='Zero Approx' if t_zero == known_zeros_im[0] else None)

ax.set_title('Enhanced Riemann Zeta Landscape: Frame-Shifted Primes and Zero Approximations')
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part (Shifted for Primes)')
ax.set_zlabel('log|ζ(s)| (Adjusted)')
ax.legend()
plt.show()