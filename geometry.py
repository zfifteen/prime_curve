"""
Advanced Prime Geometry Analysis Module

This module provides comprehensive geometric transformations and analysis tools
for prime number distributions, including curvature transformations, Fourier
analysis, Gaussian Mixture Model (GMM) clustering, dynamic origin computation,
and detailed prime gap statistics.

Author: Prime Curve Research Project
Dependencies: numpy, scipy, scikit-learn, sympy
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats
from sympy import sieve, isprime
import warnings
from typing import Tuple, List, Dict, Optional, Union

warnings.filterwarnings("ignore")

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant


class CurvatureTransform:
    """
    Advanced curvature transformation for geometric mappings of prime distributions.
    
    This class implements sophisticated transformations that map integer sequences
    into curved geometric spaces, revealing hidden patterns in prime distributions.
    """
    
    def __init__(self, base_ratio: float = PHI):
        """
        Initialize the curvature transformation.
        
        Args:
            base_ratio: The base ratio for transformations (default: golden ratio)
        """
        self.base_ratio = base_ratio
        
    def frame_shift_residues(self, n_vals: np.ndarray, k: float) -> np.ndarray:
        """
        Core curvature transformation: θ' = φ * ((n mod φ) / φ) ** k
        
        Args:
            n_vals: Array of integer values to transform
            k: Curvature exponent parameter
            
        Returns:
            Array of transformed residues
        """
        mod_phi = np.mod(n_vals, self.base_ratio) / self.base_ratio
        return self.base_ratio * np.power(mod_phi, k)
    
    def logarithmic_transform(self, n_vals: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Logarithmic curvature transformation for large-scale analysis.
        
        Args:
            n_vals: Array of integer values to transform
            scale: Scaling factor for the transformation
            
        Returns:
            Array of logarithmically transformed values
        """
        return scale * np.log(np.mod(n_vals, self.base_ratio) + 1)
    
    def hyperbolic_transform(self, n_vals: np.ndarray, hyperbolic_param: float = 1.0) -> np.ndarray:
        """
        Hyperbolic curvature transformation for non-Euclidean geometry analysis.
        
        Args:
            n_vals: Array of integer values to transform
            hyperbolic_param: Hyperbolic parameter controlling curvature
            
        Returns:
            Array of hyperbolically transformed values
        """
        normalized = np.mod(n_vals, self.base_ratio) / self.base_ratio
        return self.base_ratio * np.tanh(hyperbolic_param * normalized)
    
    def adaptive_transform(self, n_vals: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """
        Adaptive transformation that adjusts based on local density.
        
        Args:
            n_vals: Array of integer values to transform
            density_map: Local density estimates for adaptive scaling
            
        Returns:
            Array of adaptively transformed values
        """
        normalized = np.mod(n_vals, self.base_ratio) / self.base_ratio
        adaptive_k = 1.0 + density_map * 0.5  # Adaptive curvature based on density
        return self.base_ratio * np.power(normalized, adaptive_k)


class FourierAnalysis:
    """
    Spectral analysis of prime distributions using Fourier series decomposition.
    """
    
    def __init__(self, max_harmonics: int = 10):
        """
        Initialize Fourier analysis.
        
        Args:
            max_harmonics: Maximum number of harmonics to analyze
        """
        self.max_harmonics = max_harmonics
        
    def fourier_fit(self, theta_values: np.ndarray, nbins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a truncated Fourier series to the distribution.
        
        Args:
            theta_values: Array of transformed values
            nbins: Number of bins for histogram
            
        Returns:
            Tuple of (cosine coefficients, sine coefficients)
        """
        x = (theta_values % PHI) / PHI
        y, edges = np.histogram(theta_values, bins=nbins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2 / PHI
        
        # Build design matrix for Fourier series
        def design_matrix(x_vals):
            cols = [np.ones_like(x_vals)]
            for k in range(1, self.max_harmonics + 1):
                cols.append(np.cos(2 * np.pi * k * x_vals))
                cols.append(np.sin(2 * np.pi * k * x_vals))
            return np.vstack(cols).T
        
        A = design_matrix(centers)
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        
        # Separate cosine and sine coefficients
        a_coeffs = coeffs[0::2]  # cosine coefficients (including DC)
        b_coeffs = coeffs[1::2]  # sine coefficients
        
        return a_coeffs, b_coeffs
    
    def spectral_density(self, theta_values: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral density metrics.
        
        Args:
            theta_values: Array of transformed values
            
        Returns:
            Dictionary of spectral metrics
        """
        a_coeffs, b_coeffs = self.fourier_fit(theta_values)
        
        return {
            'dc_component': a_coeffs[0],
            'total_power': np.sum(a_coeffs**2) + np.sum(b_coeffs**2),
            'asymmetry_measure': np.sum(np.abs(b_coeffs)),
            'harmonic_decay_rate': self._compute_decay_rate(a_coeffs, b_coeffs),
            'dominant_frequency': self._find_dominant_frequency(a_coeffs, b_coeffs)
        }
    
    def _compute_decay_rate(self, a_coeffs: np.ndarray, b_coeffs: np.ndarray) -> float:
        """Compute the decay rate of harmonic coefficients."""
        powers = a_coeffs[1:]**2 + b_coeffs**2
        if len(powers) < 2:
            return 0.0
        
        # Fit exponential decay
        x = np.arange(1, len(powers) + 1)
        valid_idx = powers > 0
        if np.sum(valid_idx) < 2:
            return 0.0
        
        try:
            slope, _, _, _, _ = stats.linregress(x[valid_idx], np.log(powers[valid_idx]))
            return -slope
        except (ValueError, RuntimeWarning):
            return 0.0
    
    def _find_dominant_frequency(self, a_coeffs: np.ndarray, b_coeffs: np.ndarray) -> int:
        """Find the dominant frequency component."""
        powers = a_coeffs[1:]**2 + b_coeffs**2
        return np.argmax(powers) + 1 if len(powers) > 0 else 0


class GMMAnalysis:
    """
    Gaussian Mixture Model analysis for clustering in prime distributions.
    """
    
    def __init__(self, n_components: int = 5, random_state: int = 42):
        """
        Initialize GMM analysis.
        
        Args:
            n_components: Number of Gaussian components
            random_state: Random state for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        
    def fit_gmm(self, theta_values: np.ndarray) -> Tuple[GaussianMixture, Dict[str, float]]:
        """
        Fit a Gaussian Mixture Model to the distribution.
        
        Args:
            theta_values: Array of transformed values
            
        Returns:
            Tuple of (fitted GMM model, clustering metrics)
        """
        X = ((theta_values % PHI) / PHI).reshape(-1, 1)
        
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state
        ).fit(X)
        
        # Compute clustering metrics
        sigmas = np.sqrt([gmm.covariances_[i].flatten()[0] 
                         for i in range(self.n_components)])
        
        metrics = {
            'mean_sigma': np.mean(sigmas),
            'sigma_variance': np.var(sigmas),
            'cluster_separation': self._compute_separation(gmm),
            'bic_score': gmm.bic(X),
            'aic_score': gmm.aic(X),
            'log_likelihood': gmm.score(X)
        }
        
        return gmm, metrics
    
    def _compute_separation(self, gmm: GaussianMixture) -> float:
        """Compute average separation between cluster centers."""
        means = gmm.means_.flatten()
        if len(means) < 2:
            return 0.0
        
        separations = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                separations.append(abs(means[i] - means[j]))
        
        return np.mean(separations)
    
    def adaptive_components(self, theta_values: np.ndarray, 
                          max_components: int = 10) -> Tuple[int, float]:
        """
        Determine optimal number of components using information criteria.
        
        Args:
            theta_values: Array of transformed values
            max_components: Maximum number of components to test
            
        Returns:
            Tuple of (optimal n_components, best score)
        """
        X = ((theta_values % PHI) / PHI).reshape(-1, 1)
        
        scores = []
        n_range = range(1, min(max_components + 1, len(X) // 2))
        
        for n in n_range:
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',
                    random_state=self.random_state
                ).fit(X)
                scores.append(gmm.bic(X))
            except:
                scores.append(np.inf)
        
        optimal_n = n_range[np.argmin(scores)]
        best_score = min(scores)
        
        return optimal_n, best_score


class DynamicOrigin:
    """
    Dynamic origin computation for adaptive prime number geometry analysis.
    """
    
    def __init__(self):
        """Initialize dynamic origin computation."""
        pass
    
    def compute_centroid(self, theta_values: np.ndarray) -> float:
        """
        Compute the centroid of the transformed distribution.
        
        Args:
            theta_values: Array of transformed values
            
        Returns:
            Centroid position
        """
        return np.mean(theta_values % PHI)
    
    def compute_median_center(self, theta_values: np.ndarray) -> float:
        """
        Compute the median center for robust origin estimation.
        
        Args:
            theta_values: Array of transformed values
            
        Returns:
            Median center position
        """
        return np.median(theta_values % PHI)
    
    def compute_density_peak(self, theta_values: np.ndarray, nbins: int = 50) -> float:
        """
        Find the peak density location as dynamic origin.
        
        Args:
            theta_values: Array of transformed values
            nbins: Number of bins for density estimation
            
        Returns:
            Position of maximum density
        """
        hist, edges = np.histogram(theta_values % PHI, bins=nbins, density=True)
        peak_idx = np.argmax(hist)
        return (edges[peak_idx] + edges[peak_idx + 1]) / 2
    
    def adaptive_origin(self, theta_values: np.ndarray, 
                       weights: Optional[np.ndarray] = None) -> float:
        """
        Compute adaptive origin based on weighted distribution moments.
        
        Args:
            theta_values: Array of transformed values
            weights: Optional weights for each value
            
        Returns:
            Adaptive origin position
        """
        normalized_values = theta_values % PHI
        
        if weights is None:
            weights = np.ones(len(normalized_values))
        
        # Weighted centroid
        weighted_mean = np.average(normalized_values, weights=weights)
        
        # Adjust for circular nature of the domain
        circular_mean = self._circular_mean(normalized_values, weights)
        
        return circular_mean
    
    def _circular_mean(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Compute circular mean for values on [0, φ)."""
        angles = 2 * np.pi * values / PHI
        
        cos_sum = np.sum(weights * np.cos(angles))
        sin_sum = np.sum(weights * np.sin(angles))
        
        mean_angle = np.arctan2(sin_sum, cos_sum)
        
        # Convert back to [0, φ) range
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        
        return mean_angle * PHI / (2 * np.pi)


class PrimeGapStatistics:
    """
    Enhanced statistical analysis for detailed prime gap statistics.
    """
    
    def __init__(self):
        """Initialize prime gap statistics analyzer."""
        pass
    
    def compute_gap_distribution(self, primes: List[int]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute comprehensive gap distribution statistics.
        
        Args:
            primes: List of prime numbers
            
        Returns:
            Dictionary of gap statistics
        """
        if len(primes) < 2:
            return {}
        
        gaps = np.diff(primes)
        
        return {
            'mean_gap': np.mean(gaps),
            'median_gap': np.median(gaps),
            'std_gap': np.std(gaps),
            'min_gap': np.min(gaps),
            'max_gap': np.max(gaps),
            'gap_variance': np.var(gaps),
            'gap_skewness': stats.skew(gaps),
            'gap_kurtosis': stats.kurtosis(gaps),
            'unique_gaps': len(np.unique(gaps)),
            'gap_entropy': self._compute_gap_entropy(gaps),
            'gap_percentiles': np.percentile(gaps, [10, 25, 50, 75, 90, 95, 99])
        }
    
    def local_gap_analysis(self, primes: List[int], 
                          window_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Analyze local gap patterns using sliding window.
        
        Args:
            primes: List of prime numbers
            window_size: Size of sliding window
            
        Returns:
            Dictionary of local gap metrics
        """
        if len(primes) < window_size + 1:
            return {}
        
        gaps = np.diff(primes)
        n_windows = len(gaps) - window_size + 1
        
        local_means = []
        local_stds = []
        local_entropies = []
        
        for i in range(n_windows):
            window_gaps = gaps[i:i + window_size]
            local_means.append(np.mean(window_gaps))
            local_stds.append(np.std(window_gaps))
            local_entropies.append(self._compute_gap_entropy(window_gaps))
        
        return {
            'local_means': np.array(local_means),
            'local_stds': np.array(local_stds),
            'local_entropies': np.array(local_entropies),
            'mean_stability': np.std(local_means),
            'std_stability': np.std(local_stds),
            'entropy_stability': np.std(local_entropies)
        }
    
    def gap_clustering_analysis(self, primes: List[int]) -> Dict[str, float]:
        """
        Analyze clustering patterns in prime gaps.
        
        Args:
            primes: List of prime numbers
            
        Returns:
            Dictionary of clustering metrics
        """
        if len(primes) < 3:
            return {}
        
        gaps = np.diff(primes)
        
        # Consecutive gap correlations
        gap_correlation = np.corrcoef(gaps[:-1], gaps[1:])[0, 1] if len(gaps) > 1 else 0
        
        # Run length analysis for small gaps
        small_gap_runs = self._analyze_runs(gaps, threshold=np.median(gaps))
        
        # Clustering coefficient based on gap similarity
        clustering_coeff = self._compute_gap_clustering(gaps)
        
        return {
            'gap_correlation': gap_correlation,
            'small_gap_run_length': small_gap_runs['mean_run_length'],
            'clustering_coefficient': clustering_coeff,
            'gap_persistence': self._compute_persistence(gaps)
        }
    
    def _compute_gap_entropy(self, gaps: np.ndarray) -> float:
        """Compute entropy of gap distribution."""
        if len(gaps) == 0:
            return 0.0
        
        unique_gaps, counts = np.unique(gaps, return_counts=True)
        probabilities = counts / len(gaps)
        
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        return entropy
    
    def _analyze_runs(self, gaps: np.ndarray, threshold: float) -> Dict[str, float]:
        """Analyze runs of small gaps."""
        binary_sequence = gaps <= threshold
        
        runs = []
        current_run = 0
        
        for is_small in binary_sequence:
            if is_small:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        if not runs:
            return {'mean_run_length': 0.0, 'max_run_length': 0.0}
        
        return {
            'mean_run_length': np.mean(runs),
            'max_run_length': np.max(runs)
        }
    
    def _compute_gap_clustering(self, gaps: np.ndarray) -> float:
        """Compute clustering coefficient for gaps."""
        if len(gaps) < 3:
            return 0.0
        
        # Similarity based on gap magnitude
        threshold = np.std(gaps)
        clustering_sum = 0
        count = 0
        
        for i in range(1, len(gaps) - 1):
            left_similar = abs(gaps[i] - gaps[i-1]) < threshold
            right_similar = abs(gaps[i] - gaps[i+1]) < threshold
            
            if left_similar or right_similar:
                clustering_sum += int(left_similar and right_similar)
                count += 1
        
        return clustering_sum / count if count > 0 else 0.0
    
    def _compute_persistence(self, gaps: np.ndarray) -> float:
        """Compute persistence of gap patterns."""
        if len(gaps) < 2:
            return 0.0
        
        # Compute how often gap changes direction
        differences = np.diff(gaps)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        
        # Normalize by maximum possible changes
        max_changes = len(differences) - 1
        
        return 1.0 - (sign_changes / max_changes) if max_changes > 0 else 1.0


# Utility functions for integration with existing code

def generate_primes(n_max: int) -> List[int]:
    """
    Generate prime numbers up to n_max using sympy's sieve.
    
    Args:
        n_max: Maximum value for prime generation
        
    Returns:
        List of prime numbers
    """
    return list(sieve.primerange(2, n_max + 1))


def analyze_prime_geometry(primes: List[int], k_range: np.ndarray = None,
                         n_max: int = 1000) -> Dict[str, any]:
    """
    Comprehensive prime geometry analysis combining all tools.
    
    Args:
        primes: List of prime numbers
        k_range: Range of curvature parameters to test
        n_max: Maximum value for reference distribution
        
    Returns:
        Dictionary of comprehensive analysis results
    """
    if k_range is None:
        k_range = np.arange(0.1, 0.5, 0.05)
    
    # Initialize analyzers
    curve_transform = CurvatureTransform()
    fourier_analyzer = FourierAnalysis()
    gmm_analyzer = GMMAnalysis()
    origin_computer = DynamicOrigin()
    gap_analyzer = PrimeGapStatistics()
    
    # Reference distribution
    all_numbers = np.arange(1, n_max + 1)
    
    results = {
        'k_analysis': [],
        'gap_statistics': gap_analyzer.compute_gap_distribution(primes),
        'local_gap_analysis': gap_analyzer.local_gap_analysis(primes),
        'gap_clustering': gap_analyzer.gap_clustering_analysis(primes)
    }
    
    # Analyze for each k value
    for k in k_range:
        # Transform distributions
        theta_all = curve_transform.frame_shift_residues(all_numbers, k)
        theta_primes = curve_transform.frame_shift_residues(np.array(primes), k)
        
        # Fourier analysis
        fourier_metrics = fourier_analyzer.spectral_density(theta_primes)
        
        # GMM analysis
        gmm_model, gmm_metrics = gmm_analyzer.fit_gmm(theta_primes)
        
        # Dynamic origin
        adaptive_origin = origin_computer.adaptive_origin(theta_primes)
        
        k_result = {
            'k': k,
            'fourier_metrics': fourier_metrics,
            'gmm_metrics': gmm_metrics,
            'adaptive_origin': adaptive_origin
        }
        
        results['k_analysis'].append(k_result)
    
    return results