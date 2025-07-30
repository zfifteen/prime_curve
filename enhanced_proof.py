#!/usr/bin/env python3
"""
Enhanced Prime Curvature Analysis using new geometry.py module

This script demonstrates the enhanced functionality while maintaining
compatibility with the original proof.py approach.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sympy import sieve, isprime
import warnings

# Import enhanced geometry module
from geometry import (
    CurvatureTransform, FourierAnalysis, GMMAnalysis,
    DynamicOrigin, PrimeGapStatistics, analyze_prime_geometry
)

warnings.filterwarnings("ignore")

# Constants (keeping compatibility with proof.py)
phi = (1 + np.sqrt(5)) / 2
N_MAX = 1000
primes_list = list(sieve.primerange(2, N_MAX + 1))

def enhanced_proof_analysis():
    """
    Enhanced version of the proof analysis using new geometry.py functionality.
    """
    print("=== Enhanced Prime Curvature Proof with Advanced Geometry ===")
    
    # Initialize enhanced analyzers
    curve_transform = CurvatureTransform()
    fourier_analyzer = FourierAnalysis(max_harmonics=5)
    gmm_analyzer = GMMAnalysis(n_components=5)
    origin_computer = DynamicOrigin()
    gap_analyzer = PrimeGapStatistics()
    
    # High-resolution k-sweep (compatible with original)
    k_values = np.arange(0.2, 0.4001, 0.002)
    enhanced_results = []
    
    print(f"Analyzing {len(primes_list)} primes with {len(k_values)} k values...")
    
    for k in k_values:
        # Core transformation (same as original)
        theta_all = curve_transform.frame_shift_residues(np.arange(1, N_MAX + 1), k)
        theta_pr = curve_transform.frame_shift_residues(np.array(primes_list), k)
        
        # Original binning approach for compatibility
        bins = np.linspace(0, phi, 21)
        all_counts, _ = np.histogram(theta_all, bins=bins)
        pr_counts, _ = np.histogram(theta_pr, bins=bins)
        
        all_d = all_counts / len(theta_all)
        pr_d = pr_counts / len(theta_pr)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            enh = (pr_d - all_d) / all_d * 100
        
        enh = np.where(all_d > 0, enh, -np.inf)
        max_enh = np.max(enh)
        
        # Enhanced analysis using new geometry module
        fourier_metrics = fourier_analyzer.spectral_density(theta_pr)
        gmm_model, gmm_metrics = gmm_analyzer.fit_gmm(theta_pr)
        adaptive_origin = origin_computer.adaptive_origin(theta_pr)
        
        # Alternative transformations
        log_theta = curve_transform.logarithmic_transform(np.array(primes_list), scale=k)
        hyp_theta = curve_transform.hyperbolic_transform(np.array(primes_list), hyperbolic_param=k)
        
        log_fourier = fourier_analyzer.spectral_density(log_theta)
        hyp_fourier = fourier_analyzer.spectral_density(hyp_theta)
        
        enhanced_results.append({
            'k': k,
            'max_enhancement': max_enh,
            'sigma_prime': gmm_metrics['mean_sigma'],
            'fourier_b_sum': fourier_metrics['asymmetry_measure'],
            'fourier_power': fourier_metrics['total_power'],
            'adaptive_origin': adaptive_origin,
            'log_asymmetry': log_fourier['asymmetry_measure'],
            'hyp_asymmetry': hyp_fourier['asymmetry_measure'],
            'cluster_separation': gmm_metrics['cluster_separation'],
            'bic_score': gmm_metrics['bic_score']
        })
    
    # Filter valid results and find best k
    valid_results = [r for r in enhanced_results if np.isfinite(r['max_enhancement'])]
    best = max(valid_results, key=lambda r: r['max_enhancement'])
    
    # Enhanced gap analysis
    gap_stats = gap_analyzer.compute_gap_distribution(primes_list)
    local_gaps = gap_analyzer.local_gap_analysis(primes_list, window_size=50)
    gap_clustering = gap_analyzer.gap_clustering_analysis(primes_list)
    
    # Print enhanced results
    print(f"\n=== Enhanced Proof Results ===")
    print(f"Optimal curvature exponent k* = {best['k']:.3f}")
    print(f"Max mid-bin enhancement = {best['max_enhancement']:.1f}%")
    print(f"GMM σ' at k* = {best['sigma_prime']:.3f}")
    print(f"Fourier asymmetry = {best['fourier_b_sum']:.3f}")
    print(f"Spectral power = {best['fourier_power']:.1f}")
    print(f"Adaptive origin = {best['adaptive_origin']:.3f}")
    print(f"Cluster separation = {best['cluster_separation']:.3f}")
    
    print(f"\n=== Alternative Transform Results at k* ===")
    print(f"Logarithmic asymmetry = {best['log_asymmetry']:.3f}")
    print(f"Hyperbolic asymmetry = {best['hyp_asymmetry']:.3f}")
    
    print(f"\n=== Enhanced Gap Statistics ===")
    print(f"Gap entropy = {gap_stats['gap_entropy']:.3f}")
    print(f"Gap correlation = {gap_clustering['gap_correlation']:.3f}")
    print(f"Clustering coefficient = {gap_clustering['clustering_coefficient']:.3f}")
    print(f"Gap persistence = {gap_clustering['gap_persistence']:.3f}")
    
    if local_gaps:
        print(f"Mean stability = {local_gaps['mean_stability']:.3f}")
        print(f"Entropy stability = {local_gaps['entropy_stability']:.3f}")
    
    # Multi-criteria optimization
    print(f"\n=== Multi-Criteria Analysis ===")
    
    # Find k optimized for different criteria
    best_power = max(valid_results, key=lambda r: r['fourier_power'])
    best_separation = max(valid_results, key=lambda r: r['cluster_separation'])
    best_origin_stability = min(valid_results, key=lambda r: abs(r['adaptive_origin'] - np.mean([x['adaptive_origin'] for x in valid_results])))
    
    print(f"Best k for spectral power: {best_power['k']:.3f} (power: {best_power['fourier_power']:.1f})")
    print(f"Best k for cluster separation: {best_separation['k']:.3f} (sep: {best_separation['cluster_separation']:.3f})")
    print(f"Most stable adaptive origin: {best_origin_stability['k']:.3f} (origin: {best_origin_stability['adaptive_origin']:.3f})")
    
    # Statistical significance of k* selection
    enhancements = [r['max_enhancement'] for r in valid_results]
    mean_enh = np.mean(enhancements)
    std_enh = np.std(enhancements)
    z_score = (best['max_enhancement'] - mean_enh) / std_enh
    
    print(f"\n=== Statistical Significance ===")
    print(f"Mean enhancement across all k: {mean_enh:.1f}%")
    print(f"Standard deviation: {std_enh:.1f}%")
    print(f"Z-score of optimal k*: {z_score:.2f}")
    
    # Dynamic origin analysis across k values
    origins = [r['adaptive_origin'] for r in valid_results]
    origin_trend = np.polyfit(k_values[:len(origins)], origins, 1)[0]
    
    print(f"\n=== Dynamic Origin Behavior ===")
    print(f"Origin trend slope: {origin_trend:.4f}")
    print(f"Origin variance: {np.var(origins):.4f}")
    
    return best, gap_stats, local_gaps, gap_clustering

if __name__ == "__main__":
    enhanced_proof_analysis()