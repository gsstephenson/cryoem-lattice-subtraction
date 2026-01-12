"""
Core lattice subtraction algorithm.

This module contains the main LatticeSubtractor class that implements
the phase-preserving lattice removal algorithm.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

from .config import Config
from .io import read_mrc, write_mrc
from .masks import create_fft_mask, resolution_to_pixels
from .processing import (
    pad_image,
    crop_to_original,
    subtract_background,
    compute_power_spectrum,
    shift_and_average,
)


@dataclass
class SubtractionResult:
    """
    Result of lattice subtraction processing.
    
    Attributes:
        image: Processed image with lattice removed
        original_shape: Shape of input image before padding
        fft_mask: The mask used for FFT filtering (optional)
        power_spectrum: Background-subtracted power spectrum (optional)
    """
    image: np.ndarray
    original_shape: tuple
    fft_mask: Optional[np.ndarray] = None
    power_spectrum: Optional[np.ndarray] = None
    
    def save(self, path: str | Path, pixel_size: float = 1.0) -> None:
        """Save the processed image to an MRC file."""
        write_mrc(self.image, path, pixel_size=pixel_size)


class LatticeSubtractor:
    """
    Main class for lattice subtraction from cryo-EM micrographs.
    
    This class implements the algorithm from bg_push_by_rot.m:
    1. Pad image and compute 2D FFT
    2. Detect lattice peaks via thresholding on log-power spectrum
    3. Create composite mask (protect center and edges)
    4. Inpaint masked regions with local average amplitude
    5. Preserve original phase, replace amplitude
    6. Inverse FFT and crop
    
    The algorithm removes periodic crystal lattice signals while
    preserving non-periodic features in the image.
    
    Example:
        >>> config = Config(pixel_ang=0.56, threshold=1.56)
        >>> subtractor = LatticeSubtractor(config)
        >>> result = subtractor.process("input.mrc")
        >>> result.save("output.mrc")
    """
    
    def __init__(self, config: Config):
        """
        Initialize the subtractor with configuration.
        
        Args:
            config: Configuration parameters for processing
        """
        self.config = config
        self._setup_backend()
    
    def _setup_backend(self) -> None:
        """Setup computation backend (numpy or cupy)."""
        if self.config.backend == "cupy":
            try:
                import cupy as cp
                self.xp = cp
                self.use_gpu = True
            except ImportError:
                import warnings
                warnings.warn(
                    "CuPy not available, falling back to NumPy. "
                    "Install with: pip install cupy-cuda12x"
                )
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
            self.use_gpu = False
    
    def _to_device(self, array: np.ndarray) -> "np.ndarray":
        """Move array to GPU if using CuPy."""
        if self.use_gpu:
            return self.xp.asarray(array)
        return array
    
    def _to_numpy(self, array) -> np.ndarray:
        """Move array from GPU to CPU if needed."""
        if self.use_gpu:
            return self.xp.asnumpy(array)
        return array
    
    def process(
        self,
        input_path: str | Path | np.ndarray,
        return_diagnostics: bool = False,
    ) -> SubtractionResult:
        """
        Process a micrograph to remove lattice signal.
        
        Args:
            input_path: Path to input MRC file, or numpy array
            return_diagnostics: If True, include mask and power spectrum in result
            
        Returns:
            SubtractionResult containing processed image and optional diagnostics
        """
        # Load image
        if isinstance(input_path, (str, Path)):
            image = read_mrc(input_path)
        else:
            image = input_path.astype(np.float32)
        
        original_shape = image.shape
        
        # Pad image
        padded, pad_meta = pad_image(
            image,
            pad_origin=(self.config.pad_origin_y, self.config.pad_origin_x),
        )
        
        # Process
        result_padded, fft_mask, power_spec = self._process_padded(
            padded, 
            return_diagnostics=return_diagnostics,
        )
        
        # Crop to original size if requested
        if not self.config.pad_output:
            result_image = crop_to_original(result_padded, pad_meta)
        else:
            result_image = result_padded
        
        return SubtractionResult(
            image=result_image,
            original_shape=original_shape,
            fft_mask=fft_mask if return_diagnostics else None,
            power_spectrum=power_spec if return_diagnostics else None,
        )
    
    def _process_padded(
        self,
        image: np.ndarray,
        return_diagnostics: bool = False,
    ) -> tuple:
        """
        Core processing on a padded image.
        
        This implements the algorithm from bg_push_by_rot.m.
        """
        xp = self.xp
        
        # Convert to float64 for processing precision
        img = self._to_device(image.astype(np.float64))
        box_size = img.shape[0]
        
        # Step 1: Compute FFT and shift to center DC
        if self.use_gpu:
            fft_img = xp.fft.fft2(img)
            fft_shifted = xp.fft.fftshift(fft_img)
        else:
            from scipy import fft
            fft_img = fft.fft2(img)
            fft_shifted = fft.fftshift(fft_img)
        
        # Step 2: Compute log-power spectrum
        power_spectrum = xp.abs(xp.log(xp.abs(fft_shifted) + 1e-10))
        
        # Step 3: Background subtraction for peak detection
        # Move to numpy for scipy operations
        power_np = self._to_numpy(power_spectrum)
        subtracted = subtract_background(power_np)
        
        # Step 4: Threshold to detect peaks
        threshold_mask = subtracted > self.config.threshold
        
        # Step 5: Create composite mask with radial limits
        mask_final = create_fft_mask(
            box_size=box_size,
            pixel_ang=self.config.pixel_ang,
            inside_radius_ang=self.config.inside_radius_ang,
            outside_radius_ang=self.config.outside_radius_ang,
            threshold_mask=threshold_mask,
            expand_pixel=self.config.expand_pixel,
        )
        
        # Move mask to device
        mask_final = self._to_device(mask_final.astype(np.float64))
        
        # Step 6: Inpainting with local averaging
        # Keep unmasked FFT values
        fft_keep = mask_final * fft_shifted
        
        # Calculate shift distance (based on unit cell)
        shift_pixels = int(
            self.config.pixel_ang / self.config.unit_cell_ang * box_size
        )
        shift_pixels = max(1, shift_pixels)  # Ensure at least 1 pixel shift
        
        # Compute local average amplitude from unmasked regions
        amplitude_keep = xp.abs(fft_keep)
        
        # Shift and average (inpainting)
        shift_avg = (
            xp.roll(amplitude_keep, shift_pixels, axis=0) +
            xp.roll(amplitude_keep, -shift_pixels, axis=0) +
            xp.roll(amplitude_keep, shift_pixels, axis=1) +
            xp.roll(amplitude_keep, -shift_pixels, axis=1)
        ) / 4.0
        
        # Step 7: Replace masked amplitudes, preserve phase
        mask_remove = ~mask_final.astype(bool)
        inpaint_amplitude = mask_remove * shift_avg
        
        # Get original phase at masked positions
        original_phase = xp.angle(fft_shifted * mask_remove)
        
        # Reconstruct: keep + inpainted with original phase
        fft_result = fft_keep + inpaint_amplitude * xp.exp(1j * original_phase)
        
        # Step 8: Inverse FFT
        if self.use_gpu:
            fft_result = xp.fft.ifftshift(fft_result)
            result = xp.fft.ifft2(fft_result)
        else:
            from scipy import fft
            fft_result = fft.ifftshift(fft_result)
            result = fft.ifft2(fft_result)
        
        # Take real part
        result = xp.real(result).astype(xp.float32)
        
        # Move results to numpy
        result_np = self._to_numpy(result)
        mask_np = self._to_numpy(mask_final) if return_diagnostics else None
        power_np = subtracted if return_diagnostics else None
        
        return result_np, mask_np, power_np
    
    def process_array(
        self,
        image: np.ndarray,
        return_diagnostics: bool = False,
    ) -> SubtractionResult:
        """
        Process a numpy array directly.
        
        Args:
            image: Input 2D numpy array
            return_diagnostics: If True, include mask and power spectrum
            
        Returns:
            SubtractionResult
        """
        return self.process(image, return_diagnostics=return_diagnostics)


def process_micrograph(
    input_path: str | Path,
    output_path: str | Path,
    config: Config,
) -> None:
    """
    Convenience function to process a single micrograph.
    
    Args:
        input_path: Path to input MRC file
        output_path: Path for output MRC file
        config: Processing configuration
    """
    subtractor = LatticeSubtractor(config)
    result = subtractor.process(input_path)
    result.save(output_path, pixel_size=config.pixel_ang)
