#!/usr/bin/env python3
"""
helix_pipeline.py

End-to-end pipeline to:
1. Generate the first 100,000 primes.
2. Compute resonance metrics (S_bin, S_fft, combined) for each (M, k).
3. Save metrics to resonance_scores.csv.
4. Fit a 3D helix to (S_bin_norm, S_fft_norm, k).
5. Display an interactive Plotly 3D plot.

Usage:
    python helix_pipeline.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1. Generate first N primes via sieve
def gen_primes(n_primes=100_000, sieve_limit=1_500_000):
    sieve = np.ones(sieve_limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(sieve_limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i : sieve_limit+1 : i] = False
    primes = np.nonzero(sieve)[0]
    return primes[:n_primes]

# 2. Compute metrics for each modulus M and exponent k
def compute_metrics(primes, M_list, k_list, phi):
    records = []
    N = len(primes)

    for M in M_list:
        # p mod M for float M
        p_mod = np.mod(primes, M) / M  # in [0,1)
        for k in k_list:
            # map to 24-hour clock
            hours = (phi * np.power(p_mod, k)) % 24
            counts, _ = np.histogram(hours, bins=np.arange(25))
            E = N / 24.0

            # Bin-specific score: peaks at 6 and 18
            T = [6, 18]
            num = np.sum(counts[T] - E)
            den = np.sqrt(np.sum((np.delete(counts, T) - E)**2))
            S_bin = num / den

            # Fourier-based score: 4th harmonic power fraction
            F = np.fft.fft(counts)
            power = np.abs(F)
            S_fft = power[4] / np.sum(power[1:12])

            records.append({
                'M': M,
                'k': k,
                'S_bin': S_bin,
                'S_fft': S_fft
            })
    return pd.DataFrame.from_records(records)

# 3. Normalize and save
def save_metrics(df, csv_path='resonance_scores.csv'):
    df['S_bin_norm'] = df['S_bin'] / df['S_bin'].max()
    df['S_fft_norm'] = df['S_fft'] / df['S_fft'].max()
    df['combined'] = 0.5 * (df['S_bin_norm'] + df['S_fft_norm'])
    df.to_csv(csv_path, index=False)
    print(f"Saved resonance metrics to '{csv_path}'")

# 4. Fit helix parameters
def fit_helix(df):
    x = df['S_bin_norm'].values
    y = df['S_fft_norm'].values
    z = df['k'].values

    # radius a
    a = np.mean(np.sqrt(x**2 + y**2))

    # unwrapped angle
    theta = np.unwrap(np.arctan2(y, x))

    # linear fit z = m*theta + c
    m, c = np.polyfit(theta, z, 1)

    # pitch
    pitch = m * 2 * np.pi
    return a, m, c, pitch, theta

# 5. Plot interactive 3D helix
def plot_helix(df, a, m, c, theta):
    theta_fit = np.linspace(theta.min(), theta.max(), 400)
    x_fit = a * np.cos(theta_fit)
    y_fit = a * np.sin(theta_fit)
    z_fit = m * theta_fit + c

    fig = go.Figure([
        go.Scatter3d(
            x=df['S_bin_norm'],
            y=df['S_fft_norm'],
            z=df['k'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['M'],
                colorscale='Viridis',
                colorbar=dict(title='Modulus M')
            ),
            text=[f"M={M:.3f}, k={k:.3f}" for M, k in zip(df['M'], df['k'])],
            name='Data'
        ),
        go.Scatter3d(
            x=x_fit,
            y=y_fit,
            z=z_fit,
            mode='lines',
            line=dict(color='red', width=4),
            name='Fitted Helix'
        )
    ])
    fig.update_layout(
        title='3D Helix Fit to Resonance Metrics',
        scene=dict(
            xaxis_title='S_bin_norm',
            yaxis_title='S_fft_norm',
            zaxis_title='k'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()

def main():
    # constants
    phi = (1 + np.sqrt(5)) / 2
    M_list = [
        np.sqrt(2),
        np.pi,
        np.e,
        2.05,
        2.10,
        2.15,
        phi**phi
    ]
    k_list = [0.05, 0.1, 0.3, 0.5, 1.0]

    # 1
    primes = gen_primes()

    # 2
    df_metrics = compute_metrics(primes, M_list, k_list, phi)

    # 3
    save_metrics(df_metrics)

    # 4
    a, m, c, pitch, theta = fit_helix(df_metrics)
    print(f"Fitted helix radius a = {a:.4f}")
    print(f"Helix fit: z = {m:.4f}·θ + {c:.4f}")
    print(f"Helix pitch b = {pitch:.4f} (golden ratio ≈ 0.6180)")

    # 5
    plot_helix(df_metrics, a, m, c, theta)

if __name__ == '__main__':
    main()
