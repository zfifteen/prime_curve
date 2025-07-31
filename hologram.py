from abc import ABC, abstractmethod
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special  # For zeta function
from scipy import constants  # If needed, though commented

class ZetaShift(ABC):
    """
    Abstract base for ZetaShift, embodying Z = A(B/C) across domains.
    """
    def __init__(self, observed_quantity: float, rate: float, invariant: float = 299792458.0):
        self.observed_quantity = observed_quantity
        self.rate = rate
        self.INVARIANT = invariant

    @abstractmethod
    def compute_z(self) -> float:
        """Compute domain-specific Z."""
        pass

    @staticmethod
    def is_prime(num: int) -> bool:
        """Utility to check primality for flagging or adjustments."""
        if num <= 1:
            return False
        if num <= 3:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True

class NumberLineZetaShift(ZetaShift):
    """
    ZetaShift for the number line: Z = n (v_earth / c), with v_earth fixed to CMB velocity.
    Optional prime gap adjustment amplifies Z for prime resonance.
    """
    def __init__(self, n: float, rate: float = 369820.0, invariant: float = 299792458.0, use_prime_gap_adjustment: bool = False):
        super().__init__(n, rate, invariant)
        self.use_prime_gap_adjustment = use_prime_gap_adjustment

    def compute_z(self) -> float:
        base_z = self.observed_quantity * (self.rate / self.INVARIANT)
        if self.use_prime_gap_adjustment:
            gap = self._compute_prime_gap(int(self.observed_quantity))
            base_z *= (1 + gap / self.observed_quantity if self.observed_quantity != 0 else 1)
        return base_z

    def _compute_prime_gap(self, n: int) -> float:
        """Compute gap to next prime (0 if not prime)."""
        if not self.is_prime(n):
            return 0.0
        next_prime = n + 1
        while not self.is_prime(next_prime):
            next_prime += 1
        return next_prime - n

# Zeta-based transformer function
def zeta_transform(value: float, rate: float = math.e, invariant: float = math.e, use_gap: bool = False) -> float:
    shift = NumberLineZetaShift(value, rate=rate, invariant=invariant, use_prime_gap_adjustment=use_gap)
    return shift.compute_z()

# Vectorized version for arrays
vectorized_zeta = np.vectorize(zeta_transform)

# Parameters
N_POINTS = 5000
HELIX_FREQ = 0.1003033  # Tweakable ratio
LOG_SCALE = False  # Toggle for log scaling on Y-axis

# Generate data
n = np.arange(1, N_POINTS)
primality = np.vectorize(ZetaShift.is_prime)(n)  # Use ZetaShift's is_prime

# Y-values: choose raw, log, or polynomial
if LOG_SCALE:
    y_raw = np.log(n, where=(n > 1), out=np.zeros_like(n, dtype=float))
else:
    y_raw = n * (n / math.pi)

# Apply ZetaShift transform (coordinates from objects on demand)
y = vectorized_zeta(y_raw, use_gap=False)  # Set use_gap=True for prime resonance if desired

# Z-values for the helix
z = np.sin(math.pi * HELIX_FREQ * n)

# Split into primes vs non-primes
x_primes = n[primality]
y_primes = y[primality]
z_primes = z[primality]

x_nonprimes = n[~primality]
y_nonprimes = y[~primality]
z_nonprimes = z[~primality]

# Plot 1: 3D Prime Geometry Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_nonprimes, y_nonprimes, z_nonprimes, c='blue', alpha=0.3, s=10, label='Non-primes')
ax.scatter(x_primes, y_primes, z_primes, c='red', marker='*', s=50, label='Primes')
ax.set_xlabel('n (Position)')
ax.set_ylabel('Scaled Value')
ax.set_zlabel('Helical Coord')
ax.set_title('3D Prime Geometry Visualization')
ax.legend()
plt.tight_layout()
plt.show()

# Plot 2: Logarithmic spiral with prime angles
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
angle = n * 0.1 * math.pi
radius = np.log(n)
x = radius * np.cos(angle)
y = radius * np.sin(angle)
z = np.sqrt(n)  # Height shows magnitude
ax.scatter(x[~primality], y[~primality], z[~primality], c='blue', alpha=0.3, s=10)
ax.scatter(x[primality], y[primality], z[primality], c='red', marker='*', s=50, label='Primes')
ax.set_title('Prime Angles in Logarithmic Spiral')
ax.set_xlabel('X (Real)')
ax.set_ylabel('Y (Imaginary)')
ax.set_zlabel('√n (Magnitude)')
plt.show()

# Plot 3: Modular arithmetic prime clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
mod_base = 30
mod_x = n % mod_base
mod_y = (n // mod_base) % mod_base
z = np.log(n)
colors = np.where(primality, 'red', np.where(np.gcd(n, mod_base) > 1, 'purple', 'blue'))
ax.scatter(mod_x, mod_y, z, c=colors, alpha=0.7, s=15)
ax.scatter(mod_x[primality], mod_y[primality], z[primality], c='gold', marker='*', s=100, edgecolor='black', label='Primes')
ax.set_title(f'Prime Distribution mod {mod_base}')
ax.set_xlabel(f'n mod {mod_base}')
ax.set_ylabel(f'(n // {mod_base}) mod {mod_base}')
ax.set_zlabel('log(n)')
plt.show()

# Plot 4: Prime Riemann Zeta Landscape
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
real = np.linspace(0.1, 1, 100)
imag = np.linspace(10, 50, 100)
Re, Im = np.meshgrid(real, imag)
s = Re + 1j * Im
zeta_vals = np.vectorize(scipy.special.zeta)(s)
zeta_mag = np.abs(zeta_vals)
ax.plot_surface(Re, Im, np.log(zeta_mag), cmap='viridis', alpha=0.7)
prime_indices = np.where(primality)[0]
for idx in prime_indices[:300]:
    s_val = 0.5 + 1j * n[idx]
    z_val = scipy.special.zeta(s_val)
    ax.scatter(0.5, n[idx], np.log(np.abs(z_val)), c='red', s=50, edgecolor='gold')
ax.set_title('Riemann Zeta Landscape with Primes on Critical Line')
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
ax.set_zlabel('log|ζ(s)|')
plt.show()

# Plot 5: Prime Gaussian Spirals
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
angles = np.cumsum(np.where(primality, math.pi / 2, math.pi / 8))
radii = np.sqrt(n)
x = radii * np.cos(angles)
y = radii * np.sin(angles)
z = np.where(primality, np.log(n), n / (N_POINTS / 10))
ax.scatter(x[~primality], y[~primality], z[~primality], c='blue', alpha=0.4, s=15, label='Non-Primes')
ax.scatter(x[primality], y[primality], z[primality], c='red', marker='*', s=100, label='Primes')
prime_mask = primality.copy()
prime_mask[0] = False
ax.plot(x[prime_mask], y[prime_mask], z[prime_mask], 'r-', alpha=0.3)
ax.set_title('Gaussian Prime Spirals with Connection Lines')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')
plt.show()

# Plot 6: Modular Prime Torus
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
R, r = 10, 3
mod1, mod2 = 17, 23
theta = 2 * np.pi * (n % mod1) / mod1
phi = 2 * np.pi * (n % mod2) / mod2
x = (R + r * np.cos(theta)) * np.cos(phi)
y = (R + r * np.cos(theta)) * np.sin(phi)
z = r * np.sin(theta)
res_class = n % 6
colors = np.where(primality, 'red', np.where((res_class == 1) | (res_class == 5), 'blue', 'gray'))
ax.scatter(x[~primality], y[~primality], z[~primality], c=colors[~primality], alpha=0.5, s=15)
ax.scatter(x[primality], y[primality], z[primality], c='gold', marker='*', s=100, edgecolor='red')
theta_t = np.linspace(0, 2 * np.pi, 100)
phi_t = np.linspace(0, 2 * np.pi, 100)
theta_t, phi_t = np.meshgrid(theta_t, phi_t)
x_t = (R + r * np.cos(theta_t)) * np.cos(phi_t)
y_t = (R + r * np.cos(theta_t)) * np.sin(phi_t)
z_t = r * np.sin(theta_t)
ax.plot_wireframe(x_t, y_t, z_t, color='gray', alpha=0.1)
ax.set_title(f'Modular Prime Torus (Residues mod {mod1} & {mod2})')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(30, 45)
plt.show()

# Incomplete plots (as in original; add plotting if needed)
# Prime probability waves: density computed but not plotted
density = np.zeros(N_POINTS - 1)
for i in range(1, N_POINTS):
    density[i - 1] = np.sum(primality[:i]) / i

# Prime Harmonic Interference: harmonic computed but not plotted
freq1 = math.pi
freq2 = math.e
harmonic = np.sin(freq1 * n) * np.cos(freq2 * n)