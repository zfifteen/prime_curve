#!/usr/bin/env python3
"""
Demonstration of Enhanced Geometry Module Features
==================================================

This script demonstrates the four main enhancements implemented in geometry.py:
1. Golden ratio-based curvature transformations
2. Fourier analysis and Gaussian Mixture Model (GMM) integration
3. Dynamic origin computation
4. Statistical analysis enhancements

Usage: python3 demo_geometry_features.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from geometry import (CurvatureTransform, FourierAnalyzer, GMMAnalyzer,
                      DynamicOrigin, StatisticalAnalyzer, complete_geometric_analysis)

def demo_curvature_transformations():
    """Demonstrate curvature transformation capabilities."""
    print("🔄 CURVATURE TRANSFORMATIONS DEMO")
    print("-" * 40)

    # Sample geometric data - could be prime numbers, measurements, etc.
    data = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    print(f"Input data: {data}")

    ct = CurvatureTransform()

    # Golden Ratio Transformation
    golden_curved = ct.golden_ratio_transform(data, k=0.3)
    print(f"Golden ratio transformation (k=0.3): {golden_curved[:5].round(3)}...")

    # Multi-scale analysis
    scales = [0.5, 1.0, 2.0]
    multi_scale = ct.multi_scale_transform(data, scales)
    print(f"Multi-scale analysis: {len(multi_scale)} scales applied.")

    return golden_curved

def demo_fourier_gmm():
    """Demonstrate Fourier and GMM integration."""
    print("\n📊 FOURIER & GMM INTEGRATION DEMO")
    print("-" * 40)

    # Create a signal with multiple frequency components
    t = np.linspace(0, 4 * np.pi, 100)
    signal = 2 * np.sin(t) + 0.5 * np.sin(3 * t) + 0.2 * np.random.randn(len(t))

    # Fourier Analysis
    fa = FourierAnalyzer()
    fourier_result = fa.fourier_series_fit(signal, M=5)
    spectral_result = fa.spectral_analysis(signal)

    print(f"Signal length: {len(signal)}")
    print(f"Dominant frequency: {spectral_result['dominant_frequency']:.4f}")
    print(f"Spectral centroid: {spectral_result['spectral_centroid']:.4f}")
    print(f"Fourier reconstruction error: {np.mean((signal - fourier_result['reconstruction']) ** 2):.6f}")

    # GMM Analysis
    gmm = GMMAnalyzer()
    gmm_result = gmm.fit_gmm(signal)

    print(f"GMM optimal components: {gmm_result['n_components']}")
    print(f"GMM log-likelihood: {gmm_result['log_likelihood']:.4f}")

    return signal

def demo_dynamic_origin():
    """Demonstrate dynamic origin computation."""
    print("\n🎯 DYNAMIC ORIGIN COMPUTATION DEMO")
    print("-" * 40)

    # Create clustered 2D data
    np.random.seed(42)
    cluster1 = np.random.normal([2, 3], [0.5, 0.3], (30, 2))
    cluster2 = np.random.normal([5, 1], [0.3, 0.5], (20, 2))
    data_2d = np.vstack([cluster1, cluster2])

    do = DynamicOrigin()

    # Compute different types of origins
    centroid = do.compute_centroid_origin(data_2d)
    density_origin = do.compute_density_based_origin(data_2d, method='gmm')
    geometric_origin = do.compute_geometric_origin(data_2d, shape='circle')

    print(f"Centroid origin: {centroid}")
    print(f"Density origin: {density_origin}")
    print(f"Geometric origin: {geometric_origin}")

    # Adaptive origin combining methods
    criteria = {'centroid': 0.5, 'density': 0.3, 'geometric': 0.2}
    adaptive_origin = do.adaptive_origin(data_2d, criteria)
    print(f"Adaptive origin: {adaptive_origin['adaptive']}")

    return data_2d

def demo_statistical_analysis():
    """Demonstrate enhanced statistical analysis."""
    print("\n📈 STATISTICAL ANALYSIS DEMO")
    print("-" * 40)

    # Create time series data
    np.random.seed(42)
    t = np.arange(50)
    trend_data = 0.5 * t + 5 * np.sin(t / 3) + 2 * np.random.randn(len(t))

    sa = StatisticalAnalyzer()

    # Comprehensive statistics
    stats = sa.comprehensive_stats(trend_data)
    print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    print(f"Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")

    # Analyze trends
    trends = sa.trend_analysis(trend_data)
    print(f"Linear trend slope: {trends['linear_trend_slope']:.4f}")
    print(f"Volatility: {trends['volatility']:.4f}")

    return trend_data

def demo_complete_analysis():
    """Run the complete geometric analysis pipeline."""
    print("\n🚀 COMPLETE GEOMETRIC ANALYSIS DEMO")
    print("-" * 40)

    # Use prime numbers as test data
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])

    results = complete_geometric_analysis(
        primes,
        curvature_params={'k': 0.3},
        fourier_params={'M': 5},
        gmm_params={'n_components': 3},
        origin_params={'method': 'centroid'}
    )

    print("Analysis complete.")
    print(f"Curvature transformations applied: {len(results['curvature']['transformed_data'])}")
    print(f"Fourier coefficients calculated: {len(results['fourier']['series_fit']['all_coefficients'])}")
    print(f"GMM components found: {results['gmm']['fit_result']['n_components']}")

    return results

def main():
    """Run all demonstration functions."""
    print("=" * 60)
    print("ENHANCED GEOMETRY MODULE DEMONSTRATIONS")
    print("=" * 60)

    # Run demos
    demo_curvature_transformations()
    demo_fourier_gmm()
    demo_dynamic_origin()
    demo_statistical_analysis()
    demo_complete_analysis()

    print("\n✅ All demonstrations completed successfully!")

if __name__ == "__main__":
    main()