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
from .masks import create_fft_mask, create_fft_mask_gpu, resolution_to_pixels
from .processing import (
    pad_image,
    crop_to_original,
    subtract_background,
    subtract_background_gpu,
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
        threshold_used: The threshold value used (useful when threshold="auto")
    """
    image: np.ndarray
    original_shape: tuple
    fft_mask: Optional[np.ndarray] = None
    power_spectrum: Optional[np.ndarray] = None
    threshold_used: Optional[float] = None
    
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
        """Setup computation backend (numpy, pytorch, or auto).
        
        Auto mode tries PyTorch+CUDA first, then PyTorch CPU, then NumPy.
        Prints user-friendly status message about which backend is active.
        
        Uses config.device_id if specified for multi-GPU support.
        """
        backend = self.config.backend
        self._gpu_message_shown = getattr(self, '_gpu_message_shown', False)
        
        # Get device ID from config (None means auto-select GPU 0)
        device_id = self.config.device_id if self.config.device_id is not None else 0
        
        # Auto mode: try GPU first, then CPU
        if backend == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = torch.device(f'cuda:{device_id}')
                    self.use_gpu = True
                    # Only print once per session (batch processing reuses subtractor)
                    if not self._gpu_message_shown:
                        gpu_name = torch.cuda.get_device_name(device_id)
                        print(f"✓ Using GPU {device_id}: {gpu_name}")
                        self._gpu_message_shown = True
                else:
                    self.device = torch.device('cpu')
                    self.use_gpu = False
                    if not self._gpu_message_shown:
                        print("ℹ Running on CPU (run 'lattice-sub setup-gpu' to enable GPU)")
                        self._gpu_message_shown = True
            except ImportError:
                self.device = None
                self.use_gpu = False
                if not self._gpu_message_shown:
                    print("ℹ Running on CPU with NumPy (PyTorch not installed)")
                    self._gpu_message_shown = True
        
        elif backend == "pytorch":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = torch.device(f'cuda:{device_id}')
                    self.use_gpu = True
                else:
                    import warnings
                    warnings.warn(
                        "CUDA not available, falling back to CPU."
                    )
                    self.device = torch.device('cpu')
                    self.use_gpu = False
            except ImportError:
                import warnings
                warnings.warn(
                    "PyTorch not available, falling back to NumPy. "
                    "Install with: pip install torch"
                )
                self.device = None
                self.use_gpu = False
        else:
            # numpy backend
            self.device = None
            self.use_gpu = False
    
    def _to_device(self, array: np.ndarray):
        """Move array to GPU if using PyTorch."""
        if self.use_gpu and self.device is not None:
            import torch
            return torch.from_numpy(array).to(self.device)
        return array
    
    def _to_numpy(self, array) -> np.ndarray:
        """Move array from GPU to CPU if needed."""
        if self.use_gpu and hasattr(array, 'cpu'):
            return array.cpu().numpy()
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
        
        # Determine threshold - compute adaptively if "auto"
        if self.config.is_adaptive:
            from .threshold_optimizer import ThresholdOptimizer
            optimizer = ThresholdOptimizer(self.config, use_gpu=self.use_gpu)
            opt_result = optimizer.find_optimal(image)
            threshold_value = opt_result.threshold
        else:
            threshold_value = self.config.threshold
        
        # Pad image
        padded, pad_meta = pad_image(
            image,
            pad_origin=(self.config.pad_origin_y, self.config.pad_origin_x),
        )
        
        # Process
        result_padded, fft_mask, power_spec = self._process_padded(
            padded, 
            threshold=threshold_value,
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
            threshold_used=threshold_value,
        )
    
    def _process_padded(
        self,
        image: np.ndarray,
        threshold: float,
        return_diagnostics: bool = False,
    ) -> tuple:
        """
        Core processing on a padded image.
        
        This implements the algorithm from bg_push_by_rot.m.
        
        Args:
            image: Padded image array
            threshold: Peak detection threshold value
            return_diagnostics: Whether to return diagnostic arrays
        """
        # Convert to float64 for processing precision
        img = self._to_device(image.astype(np.float64))
        box_size = image.shape[0]
        
        # Step 1: Compute FFT and shift to center DC
        if self.use_gpu:
            import torch
            fft_img = torch.fft.fft2(img)
            fft_shifted = torch.fft.fftshift(fft_img)
            # Step 2: Compute log-power spectrum
            power_spectrum = torch.abs(torch.log(torch.abs(fft_shifted) + 1e-10))
        else:
            from scipy import fft
            fft_img = fft.fft2(img)
            fft_shifted = fft.fftshift(fft_img)
            # Step 2: Compute log-power spectrum
            power_spectrum = np.abs(np.log(np.abs(fft_shifted) + 1e-10))
        
        # Step 3: Background subtraction for peak detection
        # Use Kornia GPU-accelerated version if enabled, otherwise CPU scipy
        if self.use_gpu and self.config.use_kornia:
            # Keep power spectrum on GPU, use Kornia for ~50x speedup
            subtracted_tensor = subtract_background_gpu(power_spectrum)
            subtracted = self._to_numpy(subtracted_tensor)
        else:
            # Move to numpy for scipy operations
            power_np = self._to_numpy(power_spectrum)
            subtracted = subtract_background(power_np)
        
        # Step 4: Threshold to detect peaks (using passed threshold value)
        threshold_mask = subtracted > threshold
        
        # Step 5: Create composite mask with radial limits
        # Use GPU-accelerated mask creation when available
        if self.use_gpu:
            import torch
            # Convert threshold mask to GPU tensor
            threshold_tensor = torch.from_numpy(threshold_mask).to(self.device)
            
            # Create mask entirely on GPU
            mask_final_dev = create_fft_mask_gpu(
                box_size=box_size,
                pixel_ang=self.config.pixel_ang,
                inside_radius_ang=self.config.inside_radius_ang,
                outside_radius_ang=self.config.outside_radius_ang,
                threshold_mask=threshold_tensor,
                expand_pixel=self.config.expand_pixel,
                device=self.device,
            ).float()
        else:
            # CPU path
            mask_final = create_fft_mask(
                box_size=box_size,
                pixel_ang=self.config.pixel_ang,
                inside_radius_ang=self.config.inside_radius_ang,
                outside_radius_ang=self.config.outside_radius_ang,
                threshold_mask=threshold_mask,
                expand_pixel=self.config.expand_pixel,
            )
            mask_final_dev = self._to_device(mask_final.astype(np.float64))
        
        # Step 6: Inpainting with local averaging
        # Keep unmasked FFT values
        fft_keep = mask_final_dev * fft_shifted
        
        # Calculate shift distance (based on unit cell)
        shift_pixels = int(
            self.config.pixel_ang / self.config.unit_cell_ang * box_size
        )
        shift_pixels = max(1, shift_pixels)  # Ensure at least 1 pixel shift
        
        if self.use_gpu:
            import torch
            # Compute amplitude of kept FFT values (zeros where mask removes peaks)
            # This matches MATLAB: abs_y2_A = abs(y2_A) where y2_A = mask_final .* y2
            amplitude_keep = torch.abs(fft_keep)
            
            # Shift and average (inpainting) - propagates good values into zero regions
            # This matches MATLAB circshift averaging
            shift_avg = (
                torch.roll(amplitude_keep, shift_pixels, dims=0) +
                torch.roll(amplitude_keep, -shift_pixels, dims=0) +
                torch.roll(amplitude_keep, shift_pixels, dims=1) +
                torch.roll(amplitude_keep, -shift_pixels, dims=1)
            ) / 4.0
            
            # Step 7: Replace masked amplitudes, preserve ORIGINAL phase
            # MATLAB: y2_B = ~mask_final .* shift_ave
            # MATLAB: angle_y2_ori_B = angle(y .* ~mask_final)  <- uses ORIGINAL FFT phase
            mask_remove = ~mask_final_dev.bool()
            inpaint_amplitude = mask_remove.float() * shift_avg
            
            # Get original phase at masked positions FROM ORIGINAL FFT (not fft_keep)
            # This is critical - MATLAB uses: angle(y .* ~mask_final)
            original_phase = torch.angle(fft_shifted * mask_remove.float())
            
            # Reconstruct: keep + inpainted with original phase
            # MATLAB: y2 = y2_A + value_y2_B .* exp(i .* angle_y2_ori_B)
            fft_result = fft_keep + inpaint_amplitude * torch.exp(1j * original_phase)
            
            # Step 8: Inverse FFT
            fft_result = torch.fft.ifftshift(fft_result)
            result = torch.fft.ifft2(fft_result)
            
            # Take real part
            result = torch.real(result).float()
        else:
            # NumPy/SciPy path - same algorithm as GPU
            # Compute amplitude of kept FFT values (zeros where mask removes peaks)
            amplitude_keep = np.abs(fft_keep)
            
            # Shift and average (inpainting) - propagates good values into zero regions
            shift_avg = (
                np.roll(amplitude_keep, shift_pixels, axis=0) +
                np.roll(amplitude_keep, -shift_pixels, axis=0) +
                np.roll(amplitude_keep, shift_pixels, axis=1) +
                np.roll(amplitude_keep, -shift_pixels, axis=1)
            ) / 4.0
            
            # Step 7: Replace masked amplitudes, preserve ORIGINAL phase
            mask_remove = ~mask_final.astype(bool)
            inpaint_amplitude = mask_remove.astype(np.float64) * shift_avg
            
            # Get original phase at masked positions FROM ORIGINAL FFT
            original_phase = np.angle(fft_shifted * mask_remove.astype(np.float64))
            
            # Reconstruct: keep + inpainted with original phase
            fft_result = fft_keep + inpaint_amplitude * np.exp(1j * original_phase)
            
            # Step 8: Inverse FFT
            from scipy import fft
            fft_result = fft.ifftshift(fft_result)
            result = fft.ifft2(fft_result)
            
            # Take real part
            result = np.real(result).astype(np.float32)
        
        # Move results to numpy
        result_np = self._to_numpy(result)
        
        # For diagnostics, get mask as numpy array
        if return_diagnostics:
            if self.use_gpu:
                mask_np = self._to_numpy(mask_final_dev).astype(bool)
            else:
                mask_np = mask_final.astype(bool)
            power_np = subtracted
        else:
            mask_np = None
            power_np = None
        
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
