"""
Geometry Module: Enhanced Geometric Computations with Advanced Analytics
======================================================================

This module provides comprehensive geometric computation capabilities with:
1. Curvature transformations for enhanced geometric analysis
2. Fourier and GMM integration for advanced data analysis
3. Dynamic origin computation based on user-provided parameters
4. Enhanced statistical analysis with insightful metrics

All implementations are optimized for accuracy and efficiency.
"""

import numpy as np
from scipy import fft
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import warnings
from typing import Tuple, List, Dict, Optional, Union, Callable

warnings.filterwarnings("ignore")


class CurvatureTransform:
    """
    Advanced curvature transformation class for geometric computations.
    
    Supports various mathematical transformations including golden ratio,
    exponential, logarithmic, and custom user-defined transformations.
    """
    
    def __init__(self, phi: float = None):
        """Initialize with golden ratio or custom constant."""
        self.phi = phi if phi is not None else (1 + np.sqrt(5)) / 2
        self._transform_cache = {}
    
    def golden_ratio_transform(self, data: np.ndarray, k: float = 1.0, 
                             offset: float = 0.0) -> np.ndarray:
        """
        Apply golden ratio-based curvature transformation.
        
        Formula: φ * ((data mod φ + offset) / φ) ** k
        
        Args:
            data: Input data array
            k: Curvature exponent
            offset: Phase offset for transformation
            
        Returns:
            Transformed data array
        """
        cache_key = (hash(data.tobytes()), k, offset)
        if cache_key in self._transform_cache:
            return self._transform_cache[cache_key]
        
        mod_data = (np.mod(data, self.phi) + offset) / self.phi
        result = self.phi * np.power(mod_data, k)
        
        # Cache result for efficiency
        if len(self._transform_cache) < 100:  # Limit cache size
            self._transform_cache[cache_key] = result
            
        return result
    
    def exponential_curvature(self, data: np.ndarray, base: float = np.e, 
                            scale: float = 1.0) -> np.ndarray:
        """
        Apply exponential curvature transformation.
        
        Args:
            data: Input data array
            base: Exponential base (default: e)
            scale: Scaling factor
            
        Returns:
            Exponentially transformed data
        """
        return scale * np.power(base, data / np.max(data))
    
    def logarithmic_curvature(self, data: np.ndarray, base: float = np.e,
                            regularization: float = 1e-10) -> np.ndarray:
        """
        Apply logarithmic curvature transformation.
        
        Args:
            data: Input data array
            base: Logarithmic base
            regularization: Small value to prevent log(0)
            
        Returns:
            Logarithmically transformed data
        """
        safe_data = np.maximum(data, regularization)
        if base == np.e:
            return np.log(safe_data)
        else:
            return np.log(safe_data) / np.log(base)
    
    def custom_transform(self, data: np.ndarray, 
                        transform_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Apply custom user-defined transformation.
        
        Args:
            data: Input data array
            transform_func: Custom transformation function
            
        Returns:
            Transformed data using custom function
        """
        return transform_func(data)
    
    def multi_scale_transform(self, data: np.ndarray, scales: List[float], 
                            k: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Apply transformations at multiple scales.
        
        Args:
            data: Input data array
            scales: List of scale factors
            k: Curvature exponent
            
        Returns:
            Dictionary of transformed data at different scales
        """
        results = {}
        for i, scale in enumerate(scales):
            scaled_data = data * scale
            results[f'scale_{i}_{scale}'] = self.golden_ratio_transform(scaled_data, k)
        return results


class FourierAnalyzer:
    """
    Advanced Fourier analysis for geometric data with enhanced capabilities.
    """
    
    def __init__(self, max_harmonics: int = 10):
        """Initialize with maximum number of harmonics to analyze."""
        self.max_harmonics = max_harmonics
        self._fft_cache = {}
    
    def compute_fft(self, data: np.ndarray, window: str = 'hann') -> Dict[str, np.ndarray]:
        """
        Compute FFT with windowing for improved spectral analysis.
        
        Args:
            data: Input signal data
            window: Window function ('hann', 'hamming', 'blackman', 'none')
            
        Returns:
            Dictionary containing frequencies, amplitudes, and phases
        """
        cache_key = (hash(data.tobytes()), window)
        if cache_key in self._fft_cache:
            return self._fft_cache[cache_key]
        
        # Apply window function
        if window == 'hann':
            windowed_data = data * np.hanning(len(data))
        elif window == 'hamming':
            windowed_data = data * np.hamming(len(data))
        elif window == 'blackman':
            windowed_data = data * np.blackman(len(data))
        else:
            windowed_data = data
        
        # Compute FFT
        fft_result = fft.fft(windowed_data)
        frequencies = fft.fftfreq(len(data))
        amplitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        result = {
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'phases': phases,
            'complex_fft': fft_result
        }
        
        # Cache result
        if len(self._fft_cache) < 50:
            self._fft_cache[cache_key] = result
            
        return result
    
    def fourier_series_fit(self, data: np.ndarray, M: int = None, 
                          domain: Tuple[float, float] = (0, 1)) -> Dict[str, np.ndarray]:
        """
        Fit truncated Fourier series to data.
        
        Args:
            data: Input data to fit
            M: Number of harmonics (default: self.max_harmonics)
            domain: Domain for normalization
            
        Returns:
            Dictionary with Fourier coefficients and reconstruction
        """
        M = M if M is not None else self.max_harmonics
        n_points = len(data)
        
        # Normalize domain to [0, 1]
        x = np.linspace(domain[0], domain[1], n_points)
        x_norm = (x - domain[0]) / (domain[1] - domain[0])
        
        # Build design matrix for Fourier series
        design_matrix = np.ones((n_points, 2 * M + 1))
        design_matrix[:, 0] = 1  # DC component
        
        for k in range(1, M + 1):
            design_matrix[:, 2*k-1] = np.cos(2 * np.pi * k * x_norm)
            design_matrix[:, 2*k] = np.sin(2 * np.pi * k * x_norm)
        
        # Least squares fit
        coefficients, residuals, rank, s = np.linalg.lstsq(design_matrix, data, rcond=None)
        
        # Reconstruct signal
        reconstruction = design_matrix @ coefficients
        
        # Extract coefficients
        a0 = coefficients[0]
        a_coeffs = coefficients[1::2]  # Cosine coefficients
        b_coeffs = coefficients[2::2]  # Sine coefficients
        
        return {
            'a0': a0,
            'a_coefficients': a_coeffs,
            'b_coefficients': b_coeffs,
            'all_coefficients': coefficients,
            'reconstruction': reconstruction,
            'residuals': residuals,
            'x_normalized': x_norm
        }
    
    def spectral_analysis(self, data: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive spectral analysis.
        
        Args:
            data: Input signal data
            
        Returns:
            Dictionary with spectral metrics
        """
        fft_result = self.compute_fft(data)
        amplitudes = fft_result['amplitudes']
        frequencies = fft_result['frequencies']
        
        # Only consider positive frequencies
        positive_freqs = frequencies[frequencies >= 0]
        positive_amps = amplitudes[frequencies >= 0]
        
        # Calculate spectral metrics
        total_power = np.sum(positive_amps**2)
        dominant_freq_idx = np.argmax(positive_amps[1:]) + 1  # Exclude DC
        dominant_frequency = positive_freqs[dominant_freq_idx]
        dominant_amplitude = positive_amps[dominant_freq_idx]
        
        # Spectral centroid (center of mass of spectrum)
        spectral_centroid = np.sum(positive_freqs * positive_amps) / np.sum(positive_amps)
        
        # Spectral spread (variance around centroid)
        spectral_spread = np.sqrt(np.sum(((positive_freqs - spectral_centroid)**2) * positive_amps) / np.sum(positive_amps))
        
        return {
            'total_power': total_power,
            'dominant_frequency': dominant_frequency,
            'dominant_amplitude': dominant_amplitude,
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'frequency_resolution': frequencies[1] - frequencies[0]
        }


class GMMAnalyzer:
    """
    Gaussian Mixture Model analyzer for advanced clustering and density estimation.
    """
    
    def __init__(self, max_components: int = 10, random_state: int = 42):
        """Initialize GMM analyzer with parameters."""
        self.max_components = max_components
        self.random_state = random_state
        self._model_cache = {}
    
    def fit_gmm(self, data: np.ndarray, n_components: int = None, 
                covariance_type: str = 'full') -> Dict[str, Union[GaussianMixture, np.ndarray, float]]:
        """
        Fit Gaussian Mixture Model to data.
        
        Args:
            data: Input data (1D or 2D)
            n_components: Number of components (auto-selected if None)
            covariance_type: Type of covariance matrix
            
        Returns:
            Dictionary with fitted model and statistics
        """
        # Reshape 1D data for sklearn
        if data.ndim == 1:
            data_reshaped = data.reshape(-1, 1)
        else:
            data_reshaped = data
        
        # Auto-select number of components if not provided
        if n_components is None:
            n_components = self._select_optimal_components(data_reshaped)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=self.random_state
        )
        gmm.fit(data_reshaped)
        
        # Calculate additional statistics
        log_likelihood = gmm.score(data_reshaped)
        aic = gmm.aic(data_reshaped)
        bic = gmm.bic(data_reshaped)
        
        # Extract component statistics
        means = gmm.means_.flatten() if data.ndim == 1 else gmm.means_
        covariances = gmm.covariances_
        weights = gmm.weights_
        
        if data.ndim == 1:
            # For 1D data, extract standard deviations
            stds = np.sqrt(covariances.flatten())
            component_stats = {
                'means': means,
                'stds': stds,
                'weights': weights
            }
        else:
            component_stats = {
                'means': means,
                'covariances': covariances,
                'weights': weights
            }
        
        return {
            'model': gmm,
            'n_components': n_components,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'component_stats': component_stats,
            'labels': gmm.predict(data_reshaped)
        }
    
    def _select_optimal_components(self, data: np.ndarray) -> int:
        """Select optimal number of components using BIC."""
        max_comp = min(self.max_components, len(data) // 2, 20)
        
        # Ensure we have at least 1 component
        if max_comp < 1:
            return 1
            
        bic_scores = []
        
        for n in range(1, max_comp + 1):
            try:
                gmm = GaussianMixture(n_components=n, random_state=self.random_state)
                gmm.fit(data)
                bic_scores.append(gmm.bic(data))
            except:
                bic_scores.append(np.inf)
        
        if not bic_scores:
            return 1
            
        return np.argmin(bic_scores) + 1
    
    def density_estimation(self, data: np.ndarray, eval_points: np.ndarray = None) -> np.ndarray:
        """
        Estimate probability density using fitted GMM.
        
        Args:
            data: Training data
            eval_points: Points to evaluate density at (default: data points)
            
        Returns:
            Density estimates at evaluation points
        """
        gmm_result = self.fit_gmm(data)
        model = gmm_result['model']
        
        if eval_points is None:
            eval_points = data
        
        if eval_points.ndim == 1:
            eval_points = eval_points.reshape(-1, 1)
        
        return np.exp(model.score_samples(eval_points))
    
    def component_analysis(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze individual GMM components.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with component analysis results
        """
        gmm_result = self.fit_gmm(data)
        model = gmm_result['model']
        
        if data.ndim == 1:
            data_reshaped = data.reshape(-1, 1)
        else:
            data_reshaped = data
        
        # Get responsibilities (soft assignments)
        responsibilities = model.predict_proba(data_reshaped)
        
        # Calculate component contributions
        component_contributions = np.sum(responsibilities, axis=0) / len(data)
        
        # Calculate component separability (using means distance)
        means = gmm_result['component_stats']['means']
        if len(means.shape) == 1:  # 1D case
            mean_distances = np.abs(means[:, None] - means[None, :])
        else:  # Multi-dimensional case
            mean_distances = np.linalg.norm(means[:, None] - means[None, :], axis=2)
        
        return {
            'responsibilities': responsibilities,
            'component_contributions': component_contributions,
            'mean_distances': mean_distances,
            'effective_components': np.sum(component_contributions > 0.05)  # Components with >5% contribution
        }


class DynamicOrigin:
    """
    Dynamic origin computation based on user-provided parameters.
    
    Supports various origin computation methods including centroid-based,
    density-based, and custom user-defined criteria.
    """
    
    def __init__(self):
        """Initialize dynamic origin computer."""
        self._origin_cache = {}
    
    def compute_centroid_origin(self, data: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Compute centroid-based origin.
        
        Args:
            data: Input data points
            weights: Optional weights for each point
            
        Returns:
            Centroid coordinates as new origin
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if weights is None:
            return np.mean(data, axis=0)
        else:
            weights = weights / np.sum(weights)  # Normalize weights
            return np.sum(data * weights.reshape(-1, 1), axis=0)
    
    def compute_density_based_origin(self, data: np.ndarray, method: str = 'kde') -> np.ndarray:
        """
        Compute origin based on density estimation.
        
        Args:
            data: Input data points
            method: Density estimation method ('kde', 'gmm', 'histogram')
            
        Returns:
            Point of maximum density as new origin
        """
        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        
        if method == 'gmm':
            gmm_analyzer = GMMAnalyzer()
            densities = gmm_analyzer.density_estimation(data)
            max_density_idx = np.argmax(densities)
            return data[max_density_idx] if data.ndim == 1 else data[max_density_idx, :]
        
        elif method == 'histogram':
            if data.ndim == 1:
                counts, bin_edges = np.histogram(data, bins=50)
                max_bin_idx = np.argmax(counts)
                return (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
            else:
                # For 2D data, use 2D histogram
                H, x_edges, y_edges = np.histogram2d(data[:, 0], data[:, 1], bins=20)
                max_idx = np.unravel_index(np.argmax(H), H.shape)
                x_center = (x_edges[max_idx[0]] + x_edges[max_idx[0] + 1]) / 2
                y_center = (y_edges[max_idx[1]] + y_edges[max_idx[1] + 1]) / 2
                return np.array([x_center, y_center])
        
        else:  # KDE method
            from scipy.stats import gaussian_kde
            if data.ndim == 1:
                kde = gaussian_kde(data)
                x_eval = np.linspace(data.min(), data.max(), 1000)
                densities = kde(x_eval)
                return x_eval[np.argmax(densities)]
            else:
                kde = gaussian_kde(data.T)
                # Create evaluation grid
                x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 50)
                y_range = np.linspace(data[:, 1].min(), data[:, 1].max(), 50)
                X, Y = np.meshgrid(x_range, y_range)
                positions = np.vstack([X.ravel(), Y.ravel()])
                densities = kde(positions).reshape(X.shape)
                max_idx = np.unravel_index(np.argmax(densities), densities.shape)
                return np.array([X[max_idx], Y[max_idx]])
    
    def compute_geometric_origin(self, data: np.ndarray, shape: str = 'circle') -> np.ndarray:
        """
        Compute origin based on geometric fitting.
        
        Args:
            data: Input data points
            shape: Geometric shape to fit ('circle', 'ellipse', 'line')
            
        Returns:
            Geometric center as new origin
        """
        if data.ndim == 1:
            # For 1D data, return median
            return np.median(data)
        
        if shape == 'circle':
            # Fit circle and return center
            center = self._fit_circle(data)
            return center
        elif shape == 'ellipse':
            # Fit ellipse and return center
            center = self._fit_ellipse(data)
            return center
        elif shape == 'line':
            # Fit line and return midpoint
            center = self._fit_line(data)
            return center
        else:
            # Default to centroid
            return self.compute_centroid_origin(data)
    
    def _fit_circle(self, data: np.ndarray) -> np.ndarray:
        """Fit circle to 2D points and return center."""
        # Simple algebraic circle fitting
        x, y = data[:, 0], data[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2
        
        try:
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            center_x, center_y = c[0]/2, c[1]/2
            return np.array([center_x, center_y])
        except:
            return np.mean(data, axis=0)
    
    def _fit_ellipse(self, data: np.ndarray) -> np.ndarray:
        """Fit ellipse to 2D points and return center."""
        # Simplified ellipse fitting - return centroid
        return np.mean(data, axis=0)
    
    def _fit_line(self, data: np.ndarray) -> np.ndarray:
        """Fit line to points and return midpoint."""
        # Simple linear regression
        if data.shape[1] >= 2:
            x, y = data[:, 0], data[:, 1]
            # Return midpoint of data range along fitted line
            return np.array([np.mean(x), np.mean(y)])
        else:
            return np.mean(data, axis=0)
    
    def adaptive_origin(self, data: np.ndarray, criteria: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Compute adaptive origin based on multiple criteria.
        
        Args:
            data: Input data points
            criteria: Dictionary with method weights (e.g., {'centroid': 0.5, 'density': 0.3, 'geometric': 0.2})
            
        Returns:
            Dictionary with individual origins and weighted combination
        """
        origins = {}
        
        # Compute different types of origins
        if 'centroid' in criteria:
            origins['centroid'] = self.compute_centroid_origin(data)
        
        if 'density' in criteria:
            origins['density'] = self.compute_density_based_origin(data)
        
        if 'geometric' in criteria:
            origins['geometric'] = self.compute_geometric_origin(data)
        
        # Compute weighted combination
        if len(origins) > 1:
            # Normalize weights
            total_weight = sum(criteria.values())
            normalized_weights = {k: v/total_weight for k, v in criteria.items()}
            
            # Weighted average of origins
            weighted_origin = np.zeros_like(list(origins.values())[0])
            for method, origin in origins.items():
                if method in normalized_weights:
                    weighted_origin += normalized_weights[method] * origin
            
            origins['adaptive'] = weighted_origin
        
        return origins
    
    def transform_to_origin(self, data: np.ndarray, new_origin: np.ndarray) -> np.ndarray:
        """
        Transform data to new coordinate system with specified origin.
        
        Args:
            data: Input data points
            new_origin: New origin coordinates
            
        Returns:
            Transformed data relative to new origin
        """
        return data - new_origin


class StatisticalAnalyzer:
    """
    Enhanced statistical analysis with insightful metrics for geometric data.
    """
    
    def __init__(self):
        """Initialize statistical analyzer."""
        self.analysis_cache = {}
    
    def comprehensive_stats(self, data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute comprehensive statistical measures.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with comprehensive statistics
        """
        # Basic statistics
        stats = {
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'var': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data)
        }
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            stats[f'percentile_{p}'] = np.percentile(data, p)
        
        # Shape statistics
        stats['skewness'] = self._skewness(data)
        stats['kurtosis'] = self._kurtosis(data)
        
        # Robust statistics
        stats['iqr'] = stats['percentile_75'] - stats['percentile_25']
        stats['mad'] = np.median(np.abs(data - stats['median']))  # Median Absolute Deviation
        
        # Distribution tests
        stats['normality_test'] = self._test_normality(data)
        
        return stats
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, float]:
        """Test for normality using various methods."""
        from scipy import stats
        
        # Shapiro-Wilk test (for small samples)
        if len(data) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
            except:
                shapiro_stat, shapiro_p = np.nan, np.nan
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Kolmogorov-Smirnov test
        try:
            # Standardize data
            standardized = (data - np.mean(data)) / np.std(data)
            ks_stat, ks_p = stats.kstest(standardized, 'norm')
        except:
            ks_stat, ks_p = np.nan, np.nan
        
        return {
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p
        }
    
    def clustering_analysis(self, data: np.ndarray, max_clusters: int = 10) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Analyze data clustering characteristics.
        
        Args:
            data: Input data
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary with clustering analysis results
        """
        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        
        # For very small datasets, skip clustering analysis
        if len(data) < 4:
            return {
                'optimal_k_elbow': 1,
                'optimal_k_silhouette': 1,
                'best_silhouette_score': -1.0,
                'inertias': np.array([0.0]),
                'silhouette_scores': np.array([-1.0]),
                'cluster_range': np.array([1])
            }
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        max_k = min(max_clusters, len(data) // 2)
        cluster_range = range(2, max(3, max_k + 1))
        
        for k in cluster_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data_2d)
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                if len(set(labels)) > 1:
                    sil_score = silhouette_score(data_2d, labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(-1)
            except:
                inertias.append(np.inf)
                silhouette_scores.append(-1)
        
        # Find optimal k using elbow method
        if inertias:
            optimal_k = self._find_elbow(list(cluster_range), inertias)
        else:
            optimal_k = 2
        
        # Best silhouette score
        if silhouette_scores:
            best_sil_k = cluster_range[np.argmax(silhouette_scores)]
            best_sil_score = max(silhouette_scores)
        else:
            best_sil_k = 2
            best_sil_score = -1
        
        return {
            'optimal_k_elbow': optimal_k,
            'optimal_k_silhouette': best_sil_k,
            'best_silhouette_score': best_sil_score,
            'inertias': np.array(inertias),
            'silhouette_scores': np.array(silhouette_scores),
            'cluster_range': np.array(list(cluster_range))
        }
    
    def _find_elbow(self, x_vals: List[int], y_vals: List[float]) -> int:
        """Find elbow point in curve using angle method."""
        if len(x_vals) < 3:
            return x_vals[0] if x_vals else 2
        
        # Calculate angles for each point
        angles = []
        for i in range(1, len(x_vals) - 1):
            # Vectors from point i to i-1 and i+1
            v1 = np.array([x_vals[i-1] - x_vals[i], y_vals[i-1] - y_vals[i]])
            v2 = np.array([x_vals[i+1] - x_vals[i], y_vals[i+1] - y_vals[i]])
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # Find point with maximum angle (sharpest turn)
        if angles:
            elbow_idx = np.argmax(angles) + 1  # +1 because we started from index 1
            return x_vals[elbow_idx]
        else:
            return x_vals[0] if x_vals else 2
    
    def trend_analysis(self, data: np.ndarray, window_size: int = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Analyze trends in time series or sequential data.
        
        Args:
            data: Sequential data array
            window_size: Window size for moving averages
            
        Returns:
            Dictionary with trend analysis results
        """
        if window_size is None:
            window_size = max(3, len(data) // 20)
        
        # Calculate moving averages
        moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        # Calculate first and second derivatives (discrete)
        first_diff = np.diff(data)
        second_diff = np.diff(first_diff)
        
        # Linear trend (slope of best-fit line)
        x = np.arange(len(data))
        linear_trend = np.polyfit(x, data, 1)[0]
        
        # Trend strength (correlation with linear trend)
        trend_line = linear_trend * x + np.mean(data)
        trend_strength = np.corrcoef(data, trend_line)[0, 1]
        
        # Volatility measures
        volatility = np.std(first_diff)
        relative_volatility = volatility / np.mean(np.abs(data)) if np.mean(np.abs(data)) > 0 else 0
        
        return {
            'linear_trend_slope': linear_trend,
            'trend_strength': trend_strength,
            'moving_average': moving_avg,
            'first_derivative': first_diff,
            'second_derivative': second_diff,
            'volatility': volatility,
            'relative_volatility': relative_volatility,
            'monotonic_increasing': np.all(first_diff >= 0),
            'monotonic_decreasing': np.all(first_diff <= 0)
        }
    
    def correlation_analysis(self, data1: np.ndarray, data2: np.ndarray = None) -> Dict[str, float]:
        """
        Analyze correlations in data.
        
        Args:
            data1: First dataset or multivariate data
            data2: Second dataset (optional)
            
        Returns:
            Dictionary with correlation analysis results
        """
        if data2 is not None:
            # Cross-correlation between two datasets
            pearson_corr = np.corrcoef(data1, data2)[0, 1]
            
            # Spearman rank correlation
            from scipy.stats import spearmanr
            spearman_corr, spearman_p = spearmanr(data1, data2)
            
            # Lag correlation (for time series)
            max_lag = min(len(data1) // 4, 50)
            lag_correlations = []
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    lag_corr = pearson_corr
                elif lag > 0:
                    if len(data1) > lag:
                        lag_corr = np.corrcoef(data1[:-lag], data2[lag:])[0, 1]
                    else:
                        lag_corr = np.nan
                else:  # lag < 0
                    if len(data2) > -lag:
                        lag_corr = np.corrcoef(data1[-lag:], data2[:lag])[0, 1]
                    else:
                        lag_corr = np.nan
                lag_correlations.append(lag_corr)
            
            max_lag_corr = np.nanmax(np.abs(lag_correlations))
            best_lag = np.arange(-max_lag, max_lag + 1)[np.nanargmax(np.abs(lag_correlations))]
            
            return {
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'max_lag_correlation': max_lag_corr,
                'best_lag': best_lag,
                'lag_correlations': np.array(lag_correlations)
            }
        
        else:
            # Auto-correlation analysis
            from scipy.signal import correlate
            
            # Normalize data
            normalized_data = (data1 - np.mean(data1)) / np.std(data1)
            
            # Auto-correlation
            autocorr = correlate(normalized_data, normalized_data, mode='full')
            autocorr = autocorr / autocorr[len(autocorr)//2]  # Normalize by zero-lag
            
            # Find first zero crossing (decorrelation time)
            center = len(autocorr) // 2
            positive_lags = autocorr[center:]
            decorr_time = 1
            for i, val in enumerate(positive_lags[1:], 1):
                if val <= 0:
                    decorr_time = i
                    break
            
            return {
                'autocorrelation': autocorr,
                'decorrelation_time': decorr_time,
                'max_autocorr_lag': len(positive_lags) - 1
            }


# Convenience function for complete geometric analysis
def complete_geometric_analysis(data: np.ndarray, 
                              curvature_params: Dict = None,
                              fourier_params: Dict = None,
                              gmm_params: Dict = None,
                              origin_params: Dict = None) -> Dict[str, Dict]:
    """
    Perform complete geometric analysis using all available tools.
    
    Args:
        data: Input data array
        curvature_params: Parameters for curvature transformation
        fourier_params: Parameters for Fourier analysis
        gmm_params: Parameters for GMM analysis
        origin_params: Parameters for origin computation
        
    Returns:
        Dictionary with complete analysis results
    """
    results = {}
    
    # Set default parameters
    curvature_params = curvature_params or {'k': 1.0}
    fourier_params = fourier_params or {'M': 10}
    gmm_params = gmm_params or {}
    origin_params = origin_params or {'method': 'centroid'}
    
    # Curvature transformation analysis
    ct = CurvatureTransform()
    transformed_data = ct.golden_ratio_transform(data, **curvature_params)
    results['curvature'] = {
        'original_data': data,
        'transformed_data': transformed_data,
        'transformation_params': curvature_params
    }
    
    # Fourier analysis
    fa = FourierAnalyzer()
    fourier_result = fa.fourier_series_fit(data, **fourier_params)
    spectral_result = fa.spectral_analysis(data)
    results['fourier'] = {
        'series_fit': fourier_result,
        'spectral_analysis': spectral_result
    }
    
    # GMM analysis
    gmm_analyzer = GMMAnalyzer()
    gmm_result = gmm_analyzer.fit_gmm(data, **gmm_params)
    component_result = gmm_analyzer.component_analysis(data)
    results['gmm'] = {
        'fit_result': gmm_result,
        'component_analysis': component_result
    }
    
    # Dynamic origin computation
    do = DynamicOrigin()
    if origin_params.get('method') == 'adaptive':
        criteria = origin_params.get('criteria', {'centroid': 0.5, 'density': 0.5})
        origin_result = do.adaptive_origin(data, criteria)
    else:
        method = origin_params.get('method', 'centroid')
        if method == 'centroid':
            origin = do.compute_centroid_origin(data)
        elif method == 'density':
            origin = do.compute_density_based_origin(data)
        elif method == 'geometric':
            origin = do.compute_geometric_origin(data)
        else:
            origin = do.compute_centroid_origin(data)
        origin_result = {'computed_origin': origin}
    
    results['origin'] = origin_result
    
    # Statistical analysis
    sa = StatisticalAnalyzer()
    stats_result = sa.comprehensive_stats(data)
    clustering_result = sa.clustering_analysis(data)
    trend_result = sa.trend_analysis(data)
    corr_result = sa.correlation_analysis(data)
    
    results['statistics'] = {
        'comprehensive_stats': stats_result,
        'clustering_analysis': clustering_result,
        'trend_analysis': trend_result,
        'correlation_analysis': corr_result
    }
    
    return results