"""
Image processing utilities.

This module contains functions for image padding, background subtraction,
and other preprocessing operations.
"""

import numpy as np
from typing import Tuple


def pad_image(
    image: np.ndarray,
    pad_origin: Tuple[int, int],
    target_size: int | None = None,
    pad_value: float | None = None,
) -> Tuple[np.ndarray, dict]:
    """
    Pad an image with mean border for FFT processing.
    
    This replicates the MATLAB padarray functionality with 'pre' and 'post'
    padding using the image mean value.
    
    Args:
        image: Input 2D image
        pad_origin: Padding offsets (pad_y, pad_x) - pixels to add at start
        target_size: Target square size. If None, auto-calculated.
        pad_value: Value to use for padding. If None, uses image mean.
        
    Returns:
        Tuple of:
        - Padded image
        - Metadata dict with original shape and padding info for cropping
    """
    orig_h, orig_w = image.shape
    pad_y, pad_x = pad_origin
    
    # Auto-calculate target size if not provided
    if target_size is None:
        max_dim = max(orig_h, orig_w)
        target_size = max_dim + pad_x * 2
        # Round to nearest 10 for FFT efficiency
        target_size = int(np.round(target_size / 10) * 10)
    
    # Calculate padding amounts
    pad_top = pad_y - 1 if pad_y > 0 else 0
    pad_left = pad_x - 1 if pad_x > 0 else 0
    pad_bottom = target_size - orig_h - pad_top
    pad_right = target_size - orig_w - pad_left
    
    # Ensure non-negative padding
    pad_bottom = max(0, pad_bottom)
    pad_right = max(0, pad_right)
    
    # Use image mean for padding value if not specified
    if pad_value is None:
        pad_value = float(np.mean(image))
    
    # Perform padding
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=pad_value,
    )
    
    # Store metadata for later cropping
    metadata = {
        'original_shape': (orig_h, orig_w),
        'pad_top': pad_top,
        'pad_left': pad_left,
        'pad_bottom': pad_bottom,
        'pad_right': pad_right,
        'target_size': target_size,
    }
    
    return padded, metadata


def crop_to_original(
    image: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    """
    Crop a padded image back to its original size.
    
    Args:
        image: Padded image
        metadata: Metadata dict from pad_image()
        
    Returns:
        Cropped image with original dimensions
    """
    orig_h, orig_w = metadata['original_shape']
    pad_top = metadata['pad_top']
    pad_left = metadata['pad_left']
    
    return image[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w]


def subtract_background(
    image: np.ndarray,
    median_filter_size: int = 10,
) -> np.ndarray:
    """
    Subtract smooth background from an image to reveal sharp features.
    
    This is the Python equivalent of bg_FastSubtract_standard.m.
    It creates a smoothed version of the image using median filtering
    and subtracts it from the original.
    
    Args:
        image: Input 2D image (typically log-power spectrum)
        median_filter_size: Size of median filter kernel. Default: 10
        
    Returns:
        Background-subtracted image with edge regions replaced by mean
    """
    from scipy import ndimage
    from skimage.transform import resize
    
    h, w = image.shape
    
    if max(h, w) < 500:
        # For small images, apply median filter directly
        smoothed = ndimage.median_filter(image, size=median_filter_size)
        edge = median_filter_size
    else:
        # For large images, downsample -> filter -> upsample
        shrink_factor = 500 / max(h, w)
        
        # Downsample
        small = resize(
            image, 
            (int(h * shrink_factor), int(w * shrink_factor)),
            order=1,  # Bilinear
            preserve_range=True,
            anti_aliasing=True,
        )
        
        # Apply median filter
        small = ndimage.median_filter(small, size=median_filter_size)
        
        # Upsample back to original size
        smoothed = resize(
            small,
            (h, w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
        
        # Scale edge hiding region
        edge = int(median_filter_size / shrink_factor)
    
    # Subtract background
    subtracted = image - smoothed
    
    # Hide edges with artifacts
    mean_value = np.mean(subtracted)
    edge = max(1, edge)
    
    # Replace edge regions with mean
    subtracted[:edge, :] = mean_value
    subtracted[-edge:, :] = mean_value
    subtracted[:, :edge] = mean_value
    subtracted[:, -edge:] = mean_value
    
    return subtracted.astype(np.float32)


def subtract_background_gpu(
    image: "torch.Tensor",
    median_filter_size: int = 10,
) -> "torch.Tensor":
    """
    GPU-accelerated background subtraction using Kornia median filter.
    
    This is equivalent to subtract_background() but runs entirely on GPU
    using PyTorch and Kornia for ~50x speedup on large images.
    
    Args:
        image: Input 2D tensor on GPU (typically log-power spectrum)
        median_filter_size: Size of median filter kernel. Default: 10
        
    Returns:
        Background-subtracted tensor on GPU
        
    Requires:
        kornia: pip install kornia
    """
    import torch
    import torch.nn.functional as F
    import kornia
    
    device = image.device
    h, w = image.shape
    shrink_factor = 500 / max(h, w) if max(h, w) >= 500 else 1.0
    
    # Kornia expects [B, C, H, W] format
    img_4d = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    if shrink_factor < 1.0:
        small_h, small_w = int(h * shrink_factor), int(w * shrink_factor)
        small = F.interpolate(img_4d, size=(small_h, small_w), 
                              mode='bilinear', align_corners=False)
        
        # Kornia median_blur requires odd kernel size
        ks = median_filter_size if median_filter_size % 2 == 1 else median_filter_size + 1
        filtered = kornia.filters.median_blur(small, (ks, ks))
        
        smoothed = F.interpolate(filtered, size=(h, w), 
                                 mode='bilinear', align_corners=False).squeeze()
        edge = int(median_filter_size / shrink_factor)
    else:
        ks = median_filter_size if median_filter_size % 2 == 1 else median_filter_size + 1
        smoothed = kornia.filters.median_blur(img_4d, (ks, ks)).squeeze()
        edge = median_filter_size
    
    # Subtract background
    subtracted = image - smoothed
    
    # Hide edges with mean value
    mean_value = torch.mean(subtracted)
    edge = max(1, edge)
    subtracted[:edge, :] = mean_value
    subtracted[-edge:, :] = mean_value
    subtracted[:, :edge] = mean_value
    subtracted[:, -edge:] = mean_value
    
    return subtracted


def compute_power_spectrum(
    fft_shifted: np.ndarray,
    log_scale: bool = True,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute power spectrum from shifted FFT.
    
    Args:
        fft_shifted: Centered FFT (after fftshift)
        log_scale: If True, return log of amplitude. Default: True
        epsilon: Small value to avoid log(0). Default: 1e-10
        
    Returns:
        Power spectrum (log-amplitude if log_scale=True)
    """
    amplitude = np.abs(fft_shifted)
    
    if log_scale:
        return np.log(amplitude + epsilon)
    
    return amplitude


def shift_and_average(
    array: np.ndarray,
    shift_pixels: int,
) -> np.ndarray:
    """
    Create a local average by averaging 4 shifted copies.
    
    This is the inpainting technique from bg_push_by_rot.m that
    averages amplitude values from neighboring regions.
    
    Args:
        array: Input 2D array (typically FFT amplitude)
        shift_pixels: Number of pixels to shift in each direction
        
    Returns:
        Averaged array (local background estimate)
    """
    # Shift in 4 cardinal directions and average
    shifted_sum = (
        np.roll(array, shift_pixels, axis=0) +
        np.roll(array, -shift_pixels, axis=0) +
        np.roll(array, shift_pixels, axis=1) +
        np.roll(array, -shift_pixels, axis=1)
    )
    
    return shifted_sum / 4.0
