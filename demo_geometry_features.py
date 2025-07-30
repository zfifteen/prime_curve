#!/usr/bin/env python3
"""
Demonstration of Enhanced Geometry Module Features
==================================================

This script demonstrates the four main enhancements implemented in geometry.py:
1. Curvature transformations
2. Fourier and GMM integration
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
    
    # Golden ratio transformation
    golden_curved = ct.golden_ratio_transform(data, k=0.3)
    print(f"Golden ratio transform: {golden_curved[:5].round(3)}...")
    
    # Exponential curvature
    exp_curved = ct.exponential_curvature(data, scale=0.1)
    print(f"Exponential curvature: {exp_curved[:5].round(3)}...")
    
    # Multi-scale analysis
    scales = [0.5, 1.0, 2.0]
    multi_scale = ct.multi_scale_transform(data, scales)
    print(f"Multi-scale analysis: {len(multi_scale)} different scales")
    
    return golden_curved

def demo_fourier_gmm():
    """Demonstrate Fourier and GMM integration."""
    print("\n📊 FOURIER & GMM INTEGRATION DEMO")
    print("-" * 40)
    
    # Create a signal with multiple frequency components
    t = np.linspace(0, 4*np.pi, 100)
    signal = 2*np.sin(t) + 0.5*np.sin(3*t) + 0.2*np.random.randn(len(t))
    
    # Fourier analysis
    fa = FourierAnalyzer()
    fourier_result = fa.fourier_series_fit(signal, M=5)
    spectral_result = fa.spectral_analysis(signal)
    
    print(f"Signal length: {len(signal)}")
    print(f"Dominant frequency: {spectral_result['dominant_frequency']:.4f}")
    print(f"Spectral centroid: {spectral_result['spectral_centroid']:.4f}")
    print(f"Fourier reconstruction error: {np.mean((signal - fourier_result['reconstruction'])**2):.6f}")
    
    # GMM analysis
    gmm = GMMAnalyzer()
    gmm_result = gmm.fit_gmm(signal)
    density_estimates = gmm.density_estimation(signal)
    
    print(f"GMM optimal components: {gmm_result['n_components']}")
    print(f"GMM log-likelihood: {gmm_result['log_likelihood']:.2f}")
    print(f"Average density: {np.mean(density_estimates):.4f}")
    
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
    
    # Different origin computation methods
    centroid = do.compute_centroid_origin(data_2d)
    density_origin = do.compute_density_based_origin(data_2d, method='gmm')
    geometric_origin = do.compute_geometric_origin(data_2d, shape='circle')
    
    print(f"Data shape: {data_2d.shape}")
    print(f"Centroid origin: [{centroid[0]:.3f}, {centroid[1]:.3f}]")
    print(f"Density origin: [{density_origin[0]:.3f}, {density_origin[1]:.3f}]")
    print(f"Geometric origin: [{geometric_origin[0]:.3f}, {geometric_origin[1]:.3f}]")
    
    # Adaptive origin combining methods
    criteria = {'centroid': 0.5, 'density': 0.3, 'geometric': 0.2}
    adaptive_origins = do.adaptive_origin(data_2d, criteria)
    adaptive = adaptive_origins['adaptive']
    print(f"Adaptive origin: [{adaptive[0]:.3f}, {adaptive[1]:.3f}]")
    
    # Transform to new coordinate system
    transformed = do.transform_to_origin(data_2d, adaptive)
    print(f"Transformed data center: [{np.mean(transformed, axis=0)[0]:.6f}, {np.mean(transformed, axis=0)[1]:.6f}]")
    
    return data_2d, adaptive

def demo_statistical_analysis():
    """Demonstrate enhanced statistical analysis."""
    print("\n📈 STATISTICAL ANALYSIS ENHANCEMENTS DEMO")
    print("-" * 40)
    
    # Create time series with trend and noise
    np.random.seed(42)
    t = np.arange(50)
    trend_data = 0.5*t + 5*np.sin(t/3) + 2*np.random.randn(len(t))
    
    sa = StatisticalAnalyzer()
    
    # Comprehensive statistics
    stats = sa.comprehensive_stats(trend_data)
    print(f"Data points: {stats['count']}")
    print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    print(f"Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")
    print(f"IQR: {stats['iqr']:.3f}, MAD: {stats['mad']:.3f}")
    
    # Clustering analysis
    clustering = sa.clustering_analysis(trend_data)
    print(f"Optimal clusters: {clustering['optimal_k_silhouette']}")
    print(f"Silhouette score: {clustering['best_silhouette_score']:.4f}")
    
    # Trend analysis
    trends = sa.trend_analysis(trend_data)
    print(f"Linear trend slope: {trends['linear_trend_slope']:.4f}")
    print(f"Trend strength: {trends['trend_strength']:.4f}")
    print(f"Volatility: {trends['volatility']:.4f}")
    
    # Correlation analysis
    correlations = sa.correlation_analysis(trend_data)
    print(f"Decorrelation time: {correlations['decorrelation_time']}")
    
    return trend_data

def demo_complete_analysis():
    """Demonstrate the complete geometric analysis workflow."""
    print("\n🚀 COMPLETE GEOMETRIC ANALYSIS DEMO")
    print("-" * 40)
    
    # Use prime numbers as test data
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    
    # Run complete analysis with custom parameters
    results = complete_geometric_analysis(
        primes,
        curvature_params={'k': 0.3, 'offset': 0.1},
        fourier_params={'M': 5},
        gmm_params={'n_components': 3},
        origin_params={'method': 'centroid'}
    )
    
    print(f"Input: {len(primes)} prime numbers")
    print(f"Analysis components: {list(results.keys())}")
    
    # Summary of results
    print("\nAnalysis Summary:")
    print(f"• Curvature: {len(results['curvature']['transformed_data'])} transformed points")
    print(f"• Fourier: {len(results['fourier']['series_fit']['all_coefficients'])} coefficients")
    print(f"• GMM: {results['gmm']['fit_result']['n_components']} components identified")
    print(f"• Statistics: {results['statistics']['comprehensive_stats']['count']} data points analyzed")
    
    return results

def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("ENHANCED GEOMETRY MODULE DEMONSTRATION")
    print("=" * 60)
    
    # Run each demo
    curvature_data = demo_curvature_transformations()
    signal_data = demo_fourier_gmm()
    spatial_data, origin = demo_dynamic_origin()
    time_series = demo_statistical_analysis()
    complete_results = demo_complete_analysis()
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n🎉 The enhanced geometry module provides:")
    print("• Advanced curvature transformations for geometric data")
    print("• Sophisticated Fourier and GMM analysis capabilities")
    print("• Flexible dynamic origin computation methods")
    print("• Comprehensive statistical analysis with modern metrics")
    print("• Integrated workflow for complete geometric analysis")
    print("\n💡 All features are optimized for accuracy and efficiency!")

if __name__ == "__main__":
    main()