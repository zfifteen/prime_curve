import math
import numpy as np
import plotly.graph_objects as go
from sympy import isprime, divisor_count

# Universal constants from updated knowledge base
UNIVERSAL_C = math.e  # Invariant center (c analog)
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio for resonance
PI = math.pi  # For gap scaling in Z form

class WarpedNumberspace:
    """
    Simplified implementation of Z = A(B/C), where A is the observed quantity (n),
    B is the rate (prime_gap), and C is the invariant.
    """
    def __init__(self, invariant: float = UNIVERSAL_C):
        self._invariant = invariant  # Central origin (C)

    def __call__(self, n: float, prime_gap: float = 1.0) -> float:
        """
        Compute Z = A(B/C), with A = n, B = prime_gap, C = invariant.
        """
        if n <= 1:
            return 0.0
        return n * (prime_gap / self._invariant)

# Demonstration parameters from prime_number_geometry and lightprimes
N_POINTS = 100000
HELIX_FREQ = 1 / (2 * PI * PHI)  # Optimized with golden ratio for resonance

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
        if gap_index + 1 < len(primes):
            actual_gap = primes[gap_index + 1] - n
        else:
            actual_gap = primes[gap_index] - primes[gap_index - 1]  # Use last gap for the final prime
        gaps[n-1] = actual_gap / PI
        prev_prime = n
        gap_index += 1
    else:
        if gap_index < len(primes):
            next_prime = primes[gap_index]
            actual_gap = next_prime - prev_prime
        else:
            actual_gap = primes[-1] - primes[-2]  # Use last gap if beyond
        gaps[n-1] = actual_gap / PI

# Instantiate warped space
warped_space = WarpedNumberspace()

# Compute y in warped space (no pre-transform; space handles it)
y = np.array([warped_space(n, gaps[int(n)-1]) for n in n_vals])

# Compute κ(n) for curvature scaling
kappa = np.array([divisor_count(int(n)) * math.log(n + 1) / (math.e ** 2) if n > 1 else 0.0 for n in n_vals])  # Ensure float

# Z for helix, scaled by curvature to emphasize low-κ primes as peaks
z_base = np.sin(PI * HELIX_FREQ * n_vals)
z = z_base * np.exp(-kappa)  # Low kappa → higher z (peaks), high kappa → lower z (distort down)

# Split primes vs non-primes
x_primes = n_vals[primality]
y_primes = y[primality]
z_primes = z[primality]

x_nonprimes = n_vals[~primality]
y_nonprimes = y[~primality]
z_nonprimes = z[~primality]

# Subsample non-primes for performance (e.g., 10% for density)
subsample_idx = np.random.choice(len(x_nonprimes), size=int(len(x_nonprimes) * 0.1), replace=False)
x_nonprimes_sub = x_nonprimes[subsample_idx]
y_nonprimes_sub = y_nonprimes[subsample_idx]
z_nonprimes_sub = z_nonprimes[subsample_idx]

# Additional insight: Vortex filter from z_metric for efficiency
def apply_vortex_filter(numbers: np.array) -> np.array:
    """Eliminate ~71% composites via geometric constraints."""
    return numbers[(numbers > 3) & (numbers % 2 != 0) & (numbers % 3 != 0)]

filtered_primes = apply_vortex_filter(n_vals[primality])
print(f"Filtered primes: {len(filtered_primes)} out of {np.sum(primality)}")

# Visualize with Plotly for interactivity
fig = go.Figure()

# Add non-primes (subsampled)
fig.add_trace(go.Scatter3d(
    x=x_nonprimes_sub,
    y=y_nonprimes_sub,
    z=z_nonprimes_sub,
    mode='markers',
    marker=dict(size=2, color='blue', opacity=0.3),
    name='Non-primes'
))

# Add primes
fig.add_trace(go.Scatter3d(
    x=x_primes,
    y=y_primes,
    z=z_primes,
    mode='markers',
    marker=dict(size=5, color='red', symbol='circle'),
    name='Primes'
))

# Connect consecutive primes with lines
fig.add_trace(go.Scatter3d(
    x=x_primes,
    y=y_primes,
    z=z_primes,
    mode='lines',
    line=dict(color='red', width=2),
    opacity=0.5,
    name='Prime Trajectories'
))

# Update layout
fig.update_layout(
    title='Warped Prime Geometry: Invariant-Centered Space',
    scene=dict(
        xaxis_title='n (Position)',
        yaxis_title='Warped Value (Z-Transform)',
        zaxis_title='Helical Coord with Curvature Scaling'
    ),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Add custom annotation for info
info_text = f"n count: {N_POINTS}<br>Universal C: {UNIVERSAL_C:.3f}<br>Phi: {PHI:.3f}<br>Pi: {PI:.3f}<br>Helix Freq: {HELIX_FREQ:.6f}<br>Primes found: {np.sum(primality)}<br>Filtered primes: {len(filtered_primes)}"
fig.add_annotation(
    text=info_text,
    xref="paper", yref="paper",
    x=0.01, y=0.5,
    showarrow=False,
    font=dict(size=10),
    align="left",
    bgcolor="white",
    opacity=0.8
)

fig.show()