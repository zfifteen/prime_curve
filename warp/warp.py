import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import isprime, divisor_count

# Universal constants from updated knowledge base
UNIVERSAL_C = math.e  # Invariant center (c analog)
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio for resonance
PI = math.pi  # For gap scaling in Z form

class WarpedNumberspace:
    """
    Refactored Numberspace that inherently warps around the invariant C.
    Applies Z = n * (Δ_n / Δmax) directly, where Δ_n is frame shift (e.g., prime gap analog),
    and Δmax is theoretical max (scaled by π). Frame shifts emanate from C, transforming
    the number within the space for geodesic prime paths.
    Integrates curvature κ(n) = d(n) * ln(n) / e² from cognitive-number-theory.
    """
    def __init__(self, invariant: float = UNIVERSAL_C):
        self._invariant = invariant  # Central origin (C)

    def __call__(self, n: float, max_n: float, prime_gap: float = 1.0) -> float:
        """
        Transform n within the warped space: Z = n * (Δ_n / Δmax),
        where Δ_n = frame_shift(n), Δmax = π * log(max_n).
        """
        if n <= 1:
            return 0.0
        delta_n = self._compute_frame_shift(n, max_n)
        delta_max = PI * math.log(max_n + 1)  # Theoretical max gap analog
        z_transform = n * (delta_n / delta_max) * prime_gap
        kappa = self._compute_curvature(n)
        return z_transform / math.exp(kappa / self._invariant)  # Warp with curvature

    def _compute_frame_shift(self, n: float, max_n: float) -> float:
        """Frame shift from universal_frame_shift_transformer, centered on invariant."""
        base_shift = math.log(n) / math.log(max_n)
        gap_phase = 2 * PI * n / (math.log(n) + 1)
        oscillation = 0.1 * math.sin(gap_phase)
        return (base_shift + oscillation) * (1 / self._invariant)  # Emanate from C

    def _compute_curvature(self, n: float) -> float:
        """κ(n) = d(n) * ln(n) / e² from cognitive-number-theory and z_metric."""
        d_n = divisor_count(int(n))  # Use SymPy for exact divisor count
        return d_n * math.log(n) / (math.e ** 2)

# Demonstration parameters from prime_number_geometry and lightprimes
N_POINTS = 1000000
HELIX_FREQ = 0.1003033  # From main.py, tunable

# Generate data
n_vals = np.arange(1, N_POINTS + 1)
primality = np.vectorize(isprime)(n_vals)  # Use SymPy's isprime

# Instantiate warped space
warped_space = WarpedNumberspace()

# Compute y in warped space (no pre-transform; space handles it)
y = np.array([warped_space(n, N_POINTS, prime_gap=1.0 if not isprime(n) else 2.0) for n in n_vals])

# Z for helix, integrated with frame shifts
frame_shifts = np.array([warped_space._compute_frame_shift(n, N_POINTS) for n in n_vals])
z = np.sin(PI * HELIX_FREQ * n_vals) * (1 + 0.5 * frame_shifts)

# Split primes vs non-primes
x_primes = n_vals[primality]
y_primes = y[primality]
z_primes = z[primality]

x_nonprimes = n_vals[~primality]
y_nonprimes = y[~primality]
z_nonprimes = z[~primality]

# Additional insight: Vortex filter from z_metric for efficiency
def apply_vortex_filter(numbers: np.array) -> np.array:
    """Eliminate ~71% composites via geometric constraints."""
    return numbers[(numbers > 3) & (numbers % 2 != 0) & (numbers % 3 != 0)]

filtered_primes = apply_vortex_filter(n_vals[primality])
print(f"Filtered primes: {len(filtered_primes)} out of {np.sum(primality)}")

# Visualize warped geometry
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_nonprimes, y_nonprimes, z_nonprimes, c='blue', alpha=0.3, s=10, label='Non-primes')
ax.scatter(x_primes, y_primes, z_primes, c='red', marker='*', s=50, label='Primes')

ax.set_xlabel('n (Position)')
ax.set_ylabel('Warped Value (Z-Transform)')
ax.set_zlabel('Helical Coord with Frame Shifts')
ax.set_title('Warped Prime Geometry: Invariant-Centered Space')
ax.legend()

# Add custom legend on the left side
info_text = f"n count: {N_POINTS}\n"
info_text += f"Universal C: {UNIVERSAL_C:.3f}\n"
info_text += f"Phi: {PHI:.3f}\n"
info_text += f"Pi: {PI:.3f}\n"
info_text += f"Helix Freq: {HELIX_FREQ:.6f}\n"
info_text += f"Primes found: {np.sum(primality)}\n"
info_text += f"Filtered primes: {len(filtered_primes)}"

fig.text(0.01, 0.5, info_text, va='center', ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()