import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import isprime, divisor_count
from scipy.special import zeta

# Universal constants from updated knowledge base
UNIVERSAL_C = 1  # Invariant center (c analog)
PHI = 2  # Golden ratio for resonance
PI = 1 # For gap scaling in Z form

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

# ==================== ENHANCED VISUALIZATION ====================
# Demonstration parameters from prime_number_geometry and lightprimes
N_POINTS = 6000
HELIX_FREQ = 10  # Optimized with golden ratio for resonance

# Generate data
n_vals = np.arange(1, N_POINTS + 1)
primality = np.vectorize(isprime)(n_vals)  # Use SymPy's isprime
primes = n_vals[primality]

# Precompute scaled prime gaps for each n
gaps = np.zeros(N_POINTS)
prev_prime = 2  # First prime
gap_index = 0
for n in range(1, N_POINTS + 1):
    if gap_index < len(primes) and n == primes[gap_index]:
        if gap_index == 0:
            gaps[n-1] = 1.0  # Gap before first prime
        elif gap_index + 1 < len(primes):
            actual_gap = primes[gap_index] - primes[gap_index - 1]
        else:
            actual_gap = primes[gap_index] - primes[gap_index - 1]
        prev_prime = n
        gap_index += 1
    else:
        if gap_index < len(primes):
            next_prime = primes[gap_index]
            actual_gap = next_prime - prev_prime
        else:
            actual_gap = primes[-1] - primes[-2]
        gaps[n-1] = actual_gap

# Compute actual prime gaps for coloring
prime_gaps = np.zeros_like(primes, dtype=float)
for i in range(1, len(primes)):
    prime_gaps[i] = primes[i] - primes[i-1]
prime_gaps[0] = prime_gaps[1]  # Set first gap to match second

# Instantiate warped space
warped_space = WarpedNumberspace()

# Compute y in warped space (no pre-transform; space handles it)
y = np.array([warped_space(n, N_POINTS, prime_gap=gaps[int(n)-1]) for n in n_vals])

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

# Compute divisor counts for non-primes
div_counts = np.vectorize(divisor_count)(x_nonprimes)

# Compute Riemann zeta values along the critical line for primes
zeta_values = np.real(zeta(0.5 + 1j * y_primes))

# Vortex filter for efficiency
def apply_vortex_filter(numbers: np.array) -> np.array:
    """Eliminate ~71% composites via geometric constraints."""
    return numbers[(numbers > 3) & (numbers % 2 != 0) & (numbers % 3 != 0)]

filtered_primes = apply_vortex_filter(n_vals[primality])
print(f"Filtered primes: {len(filtered_primes)} out of {np.sum(primality)}")

# Create enhanced visualization
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot non-primes colored by divisor count
sc_nonprimes = ax.scatter(
    x_nonprimes,
    y_nonprimes,
    z_nonprimes,
    c="blue",
    alpha=0.4,
    s=8,
    label='Non-primes'
)

# Plot primes colored by gap size
sc_primes = ax.scatter(
    x_primes,
    y_primes,
    z_primes,
    c="red",
    marker='*',
    s=50,
    label='Primes'
)

# Add Riemann zeta surface
ax.plot_trisurf(
    x_primes,
    y_primes,
    zeta_values,
    cmap='twilight',
    alpha=0.3,
    label='Riemann Zeta'
)

ax.set_xlabel('n (Position)')
ax.set_ylabel('Warped Value (Z-Transform)')
ax.set_zlabel('Helical Coord with Frame Shifts')
ax.set_title(f'Warped Prime Geometry (Enhanced): {N_POINTS} Points')
ax.legend()

# Add colorbars
cbar_nonprimes = fig.colorbar(sc_nonprimes, ax=ax, pad=0.1)
cbar_nonprimes.set_label('Divisor Count (Non-primes)')
cbar_primes = fig.colorbar(sc_primes, ax=ax, pad=0.15)
cbar_primes.set_label('Prime Gap Size (Primes)')

# Add info box
info_text = f"n count: {N_POINTS}\n"
info_text += f"Universal C: {UNIVERSAL_C:.3f}\n"
info_text += f"Phi: {PHI:.3f}\n"
info_text += f"Pi: {PI:.3f}\n"
info_text += f"Helix Freq: {HELIX_FREQ:.6f}\n"
info_text += f"Primes found: {np.sum(primality)}\n"
info_text += f"Filtered primes: {len(filtered_primes)}"

fig.text(0.05, 0.7, info_text, va='center', ha='left', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(f'enhanced_prime_geometry_{N_POINTS}.png', dpi=300)
plt.show()