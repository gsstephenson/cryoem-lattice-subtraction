"""
Adaptive threshold optimization for lattice subtraction.

This module provides GPU-accelerated methods for determining the optimal
peak detection threshold on a per-image basis within the empirically
validated range of [1.4, 1.6].

The algorithm uses Golden Section Search with a physics-informed quality
metric that balances lattice peak removal with signal preservation.

Example:
    >>> from lattice_subtraction import ThresholdOptimizer, Config
    >>> config = Config(pixel_ang=0.56, threshold="auto")
    >>> optimizer = ThresholdOptimizer(config)
    >>> optimal_threshold = optimizer.find_optimal(image)
    >>> print(f"Optimal threshold: {optimal_threshold:.3f}")
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np

# Try to import torch for GPU operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Golden ratio for optimization
PHI = (1 + np.sqrt(5)) / 2
RESPHI = 2 - PHI  # 1 / phi


@dataclass
class OptimizationResult:
    """
    Result of threshold optimization.
    
    Attributes:
        threshold: The optimal threshold value found
        quality_score: Quality metric at optimal threshold
        iterations: Number of iterations to converge
        search_range: The [min, max] range searched
        peak_count: Number of peaks detected at optimal threshold
    """
    threshold: float
    quality_score: float
    iterations: int
    search_range: Tuple[float, float]
    peak_count: int


class ThresholdOptimizer:
    """
    GPU-accelerated threshold optimizer using Golden Section Search.
    
    This class finds the optimal threshold for lattice peak detection
    within the validated range [1.4, 1.6] by maximizing a quality metric
    that balances peak removal with signal preservation.
    
    The quality metric is based on:
    1. Peak distinctiveness: Peaks should be clearly above background
    2. Peak count stability: Threshold should be at a stable plateau
    3. Power distribution: Optimal separation of lattice vs non-lattice
    
    Example:
        >>> optimizer = ThresholdOptimizer(config)
        >>> result = optimizer.find_optimal(image)
        >>> config.threshold = result.threshold
    """
    
    # Validated threshold range (per empirical observations)
    DEFAULT_MIN_THRESHOLD = 1.40
    DEFAULT_MAX_THRESHOLD = 1.60
    
    def __init__(
        self,
        config,
        min_threshold: float = DEFAULT_MIN_THRESHOLD,
        max_threshold: float = DEFAULT_MAX_THRESHOLD,
        tolerance: float = 0.005,
        use_gpu: bool = True,
    ):
        """
        Initialize the optimizer.
        
        Args:
            config: Config object with pixel_ang and resolution parameters
            min_threshold: Minimum threshold to search (default: 1.4)
            max_threshold: Maximum threshold to search (default: 1.6)
            tolerance: Convergence tolerance (default: 0.005)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.config = config
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.tolerance = tolerance
        
        # Setup device
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                self.use_gpu = False
        else:
            self.device = None
    
    def _prepare_fft_data(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute FFT and background-subtracted power spectrum.
        
        Returns:
            Tuple of (background_subtracted_spectrum, radial_mask, box_size)
        """
        from .processing import pad_image, subtract_background
        from .masks import create_radial_band_mask, resolution_to_pixels
        
        # Pad image
        padded, _ = pad_image(
            image,
            pad_origin=(self.config.pad_origin_y, self.config.pad_origin_x),
        )
        box_size = padded.shape[0]
        
        if self.use_gpu:
            # GPU path
            img_tensor = torch.from_numpy(padded.astype(np.float64)).to(self.device)
            fft_img = torch.fft.fft2(img_tensor)
            fft_shifted = torch.fft.fftshift(fft_img)
            power_spectrum = torch.abs(torch.log(torch.abs(fft_shifted) + 1e-10))
            power_np = power_spectrum.cpu().numpy()
        else:
            # CPU path
            from scipy import fft
            fft_img = fft.fft2(padded.astype(np.float64))
            fft_shifted = fft.fftshift(fft_img)
            power_np = np.abs(np.log(np.abs(fft_shifted) + 1e-10))
        
        # Background subtraction
        subtracted = subtract_background(power_np)
        
        # Create radial band mask for valid detection region
        inner_radius = resolution_to_pixels(
            self.config.inside_radius_ang,
            self.config.pixel_ang,
            box_size,
        )
        outer_radius = resolution_to_pixels(
            self.config.outside_radius_ang,
            self.config.pixel_ang,
            box_size,
        )
        radial_mask = create_radial_band_mask(
            (box_size, box_size),
            inner_radius,
            outer_radius,
        )
        
        return subtracted, radial_mask, box_size
    
    def _compute_quality(
        self,
        subtracted: np.ndarray,
        radial_mask: np.ndarray,
        threshold: float,
    ) -> Tuple[float, int]:
        """
        Compute quality metric for a given threshold.
        
        The quality metric is designed to find the "elbow" in the peak count
        curve - where we transition from detecting noise to detecting only
        true lattice peaks.
        
        The metric combines:
        1. Peak-to-background separation: True peaks should be well separated
        2. Peak clustering: Real lattice peaks are periodic, not random noise
        3. Balanced peak count: Not too many (noise) or too few (missing peaks)
        
        Higher quality = better threshold choice.
        
        Args:
            subtracted: Background-subtracted power spectrum
            radial_mask: Mask for valid detection region
            threshold: Threshold value to evaluate
            
        Returns:
            Tuple of (quality_score, peak_count)
        """
        # Apply threshold within valid region
        peaks = (subtracted > threshold) & radial_mask
        peak_count = np.sum(peaks)
        
        if peak_count == 0:
            # No peaks detected - threshold too high
            return 0.0, 0
        
        # Get values at peak locations
        peak_values = subtracted[peaks]
        
        # Metric 1: Peak significance - mean excess above threshold
        mean_excess = np.mean(peak_values - threshold)
        
        # Metric 2: Signal-to-noise of detected peaks
        # Higher values = more confident detections
        peak_snr = np.mean(peak_values) / (np.std(peak_values) + 1e-6)
        
        # Metric 3: Peak count penalty
        # We expect ~400-1000 true lattice peaks for typical images
        # This creates a unimodal quality function
        # Too few peaks: missing real lattice
        # Too many peaks: detecting noise
        target_peaks = 600  # Approximate expected peak count
        peak_penalty = np.exp(-0.5 * ((peak_count - target_peaks) / 300) ** 2)
        
        # Metric 4: Coverage ratio - lattice typically covers 0.5-2% of FFT
        coverage = peak_count / np.sum(radial_mask)
        coverage_score = np.exp(-50 * (coverage - 0.01) ** 2)  # Peak at 1% coverage
        
        # Combined quality: balance all factors
        quality = (1 + mean_excess) * (1 + peak_snr) * peak_penalty * coverage_score
        
        return float(quality), int(peak_count)
    
    def _compute_quality_batch_gpu(
        self,
        subtracted: np.ndarray,
        radial_mask: np.ndarray,
        thresholds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute quality metrics for multiple thresholds in parallel on GPU.
        
        This is much faster than calling _compute_quality in a loop because
        we evaluate all thresholds simultaneously using tensor operations.
        
        Args:
            subtracted: Background-subtracted power spectrum
            radial_mask: Mask for valid detection region
            thresholds: Array of threshold values to evaluate
            
        Returns:
            Tuple of (quality_scores, peak_counts) arrays
        """
        import torch
        
        # Move data to GPU
        sub_tensor = torch.from_numpy(subtracted.astype(np.float32)).to(self.device)
        mask_tensor = torch.from_numpy(radial_mask.astype(np.float32)).to(self.device)
        thresh_tensor = torch.from_numpy(thresholds.astype(np.float32)).to(self.device)
        
        n_thresholds = len(thresholds)
        
        # Expand dimensions for broadcasting: (H, W) -> (1, H, W)
        sub_expanded = sub_tensor.unsqueeze(0)  # (1, H, W)
        mask_expanded = mask_tensor.unsqueeze(0)  # (1, H, W)
        thresh_expanded = thresh_tensor.view(-1, 1, 1)  # (N, 1, 1)
        
        # Compute peak masks for all thresholds at once: (N, H, W)
        peaks_all = (sub_expanded > thresh_expanded) & (mask_expanded > 0.5)
        
        # Count peaks for each threshold
        peak_counts = peaks_all.sum(dim=(1, 2))  # (N,)
        
        # Pre-compute constants
        target_peaks = 600.0
        total_mask_pixels = mask_tensor.sum()
        
        # Initialize quality scores
        qualities = torch.zeros(n_thresholds, device=self.device)
        
        # For each threshold, compute quality metrics
        # Note: We need a loop here because peak statistics vary per threshold
        for i in range(n_thresholds):
            peaks_i = peaks_all[i]
            count = peak_counts[i].item()
            
            if count == 0:
                qualities[i] = 0.0
                continue
            
            # Get peak values
            peak_values = sub_tensor[peaks_i]
            
            # Metric 1: Mean excess
            mean_excess = (peak_values - thresh_tensor[i]).mean()
            
            # Metric 2: SNR
            peak_snr = peak_values.mean() / (peak_values.std() + 1e-6)
            
            # Metric 3: Peak penalty (Gaussian around target) - use tensor for exp
            count_tensor = torch.tensor(count, dtype=torch.float32, device=self.device)
            peak_penalty = torch.exp(-0.5 * ((count_tensor - target_peaks) / 300) ** 2)
            
            # Metric 4: Coverage score
            coverage = count_tensor / total_mask_pixels
            coverage_score = torch.exp(-50 * (coverage - 0.01) ** 2)
            
            # Combined quality
            qualities[i] = (1 + mean_excess) * (1 + peak_snr) * peak_penalty * coverage_score
        
        return qualities.cpu().numpy(), peak_counts.cpu().numpy().astype(int)
    
    def _grid_search(
        self,
        subtracted: np.ndarray,
        radial_mask: np.ndarray,
    ) -> Tuple[float, float, int, int]:
        """
        Perform grid search to find optimal threshold.
        
        Uses GPU-parallel evaluation when available for speed.
        Falls back to sequential CPU evaluation otherwise.
        
        Returns:
            Tuple of (optimal_threshold, quality, iterations, peak_count)
        """
        # Evaluate at 21 points from 1.40 to 1.60 (step = 0.01)
        n_points = 21
        thresholds = np.linspace(self.min_threshold, self.max_threshold, n_points)
        
        # Use GPU batch evaluation if available
        if self.use_gpu and TORCH_AVAILABLE:
            qualities, peak_counts = self._compute_quality_batch_gpu(
                subtracted, radial_mask, thresholds
            )
            best_idx = np.argmax(qualities)
            return thresholds[best_idx], qualities[best_idx], n_points, peak_counts[best_idx]
        
        # CPU fallback: sequential evaluation
        best_threshold = self.min_threshold
        best_quality = -1
        best_peaks = 0
        
        for t in thresholds:
            quality, peaks = self._compute_quality(subtracted, radial_mask, t)
            if quality > best_quality:
                best_quality = quality
                best_threshold = t
                best_peaks = peaks
        
        return best_threshold, best_quality, n_points, best_peaks
    
    def find_optimal(
        self,
        image: np.ndarray,
    ) -> OptimizationResult:
        """
        Find the optimal threshold for the given image.
        
        This method:
        1. Computes the FFT and background-subtracted power spectrum
        2. Uses Golden Section Search to find the threshold that
           maximizes the quality metric within [1.4, 1.6]
        3. Returns the optimal threshold and diagnostics
        
        Args:
            image: Input 2D image array
            
        Returns:
            OptimizationResult with optimal threshold and metrics
        """
        # Prepare FFT data
        subtracted, radial_mask, box_size = self._prepare_fft_data(image)
        
        # Run optimization using grid search
        threshold, quality, iterations, peak_count = self._grid_search(
            subtracted, radial_mask
        )
        
        return OptimizationResult(
            threshold=threshold,
            quality_score=quality,
            iterations=iterations,
            search_range=(self.min_threshold, self.max_threshold),
            peak_count=peak_count,
        )
    
    def evaluate_threshold(
        self,
        image: np.ndarray,
        threshold: float,
    ) -> Tuple[float, int]:
        """
        Evaluate the quality of a specific threshold for an image.
        
        Useful for comparing fixed vs optimized thresholds.
        
        Args:
            image: Input 2D image array
            threshold: Threshold value to evaluate
            
        Returns:
            Tuple of (quality_score, peak_count)
        """
        subtracted, radial_mask, _ = self._prepare_fft_data(image)
        return self._compute_quality(subtracted, radial_mask, threshold)


def find_optimal_threshold(
    image: np.ndarray,
    config,
    min_threshold: float = 1.40,
    max_threshold: float = 1.60,
) -> float:
    """
    Convenience function to find optimal threshold for an image.
    
    Args:
        image: Input 2D image array
        config: Config object with pixel_ang and resolution parameters
        min_threshold: Minimum threshold to search
        max_threshold: Maximum threshold to search
        
    Returns:
        Optimal threshold value
        
    Example:
        >>> threshold = find_optimal_threshold(image, config)
        >>> config.threshold = threshold
        >>> subtractor = LatticeSubtractor(config)
        >>> result = subtractor.process(image)
    """
    optimizer = ThresholdOptimizer(
        config,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )
    result = optimizer.find_optimal(image)
    return result.threshold
