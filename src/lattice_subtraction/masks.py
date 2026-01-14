"""
Mask generation utilities for FFT processing.

This module provides functions for creating circular masks and 
performing morphological operations on masks.

GPU-accelerated versions are available when PyTorch with CUDA is present.
"""

import numpy as np
from typing import Tuple, Optional, Union

# Try to import torch for GPU operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_circular_mask(
    shape: Tuple[int, int],
    radius: float,
    center: Tuple[float, float] | None = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Create a circular binary mask.
    
    This is the Python equivalent of bg_drill_hole.m, but optimized using
    vectorized NumPy operations instead of nested loops.
    
    Args:
        shape: Output mask shape (height, width)
        radius: Radius of the circular region in pixels
        center: Center coordinates (y, x). If None, uses image center.
        invert: If True, mask is 0 inside circle, 1 outside. Default: False
        
    Returns:
        Boolean mask array where True indicates the circular region
        (or its complement if invert=True)
        
    Example:
        >>> mask = create_circular_mask((100, 100), radius=30)
        >>> mask.shape
        (100, 100)
    """
    h, w = shape
    
    if center is None:
        center = (h // 2, w // 2)
    
    cy, cx = center
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from center
    dist_sq = (y - cy) ** 2 + (x - cx) ** 2
    
    # Create mask
    mask = dist_sq < radius ** 2
    
    if invert:
        mask = ~mask
    
    return mask


def create_radial_band_mask(
    shape: Tuple[int, int],
    inner_radius: float,
    outer_radius: float,
    center: Tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Create an annular (ring) mask between two radii.
    
    Args:
        shape: Output mask shape (height, width)
        inner_radius: Inner radius of the ring
        outer_radius: Outer radius of the ring
        center: Center coordinates. If None, uses image center.
        
    Returns:
        Boolean mask that is True in the annular region
    """
    inner = create_circular_mask(shape, inner_radius, center)
    outer = create_circular_mask(shape, outer_radius, center)
    
    return outer & ~inner


def resolution_to_pixels(
    resolution_ang: float,
    pixel_size_ang: float,
    box_size: int,
) -> float:
    """
    Convert resolution in Angstroms to radius in Fourier pixels.
    
    The relationship is: radius_pixels = (pixel_size / resolution) * box_size
    
    Args:
        resolution_ang: Resolution in Angstroms
        pixel_size_ang: Pixel size in Angstroms
        box_size: Size of the FFT box
        
    Returns:
        Radius in Fourier pixels corresponding to the resolution
    """
    return (pixel_size_ang / resolution_ang) * box_size


def dilate_mask(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Dilate a binary mask using a circular structuring element.
    
    This replicates the MATLAB filter2(circle, mask) approach using
    scipy's ndimage for better performance.
    
    Args:
        mask: Input binary mask
        radius: Dilation radius in pixels
        
    Returns:
        Dilated mask
    """
    from scipy import ndimage
    
    # Create circular structuring element
    size = radius * 2 + 1
    struct = create_circular_mask((size, size), radius - 1)
    
    # Perform dilation
    dilated = ndimage.binary_dilation(mask, structure=struct)
    
    return dilated


def erode_mask(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Erode a binary mask using a circular structuring element.
    
    Args:
        mask: Input binary mask
        radius: Erosion radius in pixels
        
    Returns:
        Eroded mask
    """
    from scipy import ndimage
    
    size = radius * 2 + 1
    struct = create_circular_mask((size, size), radius - 1)
    
    eroded = ndimage.binary_erosion(mask, structure=struct)
    
    return eroded


def create_fft_mask(
    box_size: int,
    pixel_ang: float,
    inside_radius_ang: float,
    outside_radius_ang: float,
    threshold_mask: np.ndarray,
    expand_pixel: int = 10,
) -> np.ndarray:
    """
    Create the composite FFT mask for lattice spot removal.
    
    This combines:
    1. The threshold-based peak detection mask
    2. Central protection zone (low frequencies)
    3. Outer protection zone (near-Nyquist)
    4. Morphological expansion for smooth transitions
    
    Args:
        box_size: Size of the FFT (square)
        pixel_ang: Pixel size in Angstroms
        inside_radius_ang: Inner resolution limit (protect center)
        outside_radius_ang: Outer resolution limit (protect edges)
        threshold_mask: Boolean mask from peak thresholding (True = peak)
        expand_pixel: Expansion radius for morphological dilation
        
    Returns:
        Final mask where True = keep, False = replace with inpainted values
    """
    # Convert resolution to Fourier pixels
    inner_radius = resolution_to_pixels(inside_radius_ang, pixel_ang, box_size)
    outer_radius = resolution_to_pixels(outside_radius_ang, pixel_ang, box_size)
    
    # Clamp outer radius to valid range
    outer_radius = min(outer_radius, box_size // 2 - 1)
    
    # Create radial masks
    shape = (box_size, box_size)
    mask_center = create_circular_mask(shape, inner_radius)  # Protect center
    mask_outside = create_circular_mask(shape, outer_radius)  # Within processing region
    
    # Combine masks:
    # - Keep center (low freq) 
    # - Keep outside Nyquist limit
    # - In between: remove peaks (where threshold_mask is True)
    # mask = ~threshold_mask OR ~mask_outside OR mask_center
    combined = ~threshold_mask | ~mask_outside | mask_center
    
    # Any non-zero value means "keep this pixel"
    mask_final = combined
    
    # Expand the removal regions (invert, dilate, invert back)
    if expand_pixel > 0:
        # Regions to remove (inverted mask)
        removal_regions = ~mask_final
        
        # Dilate the removal regions
        rad_expand = expand_pixel // 2 - 1
        if rad_expand > 0:
            removal_regions = dilate_mask(removal_regions, rad_expand)
        
        mask_final = ~removal_regions
    
    return mask_final


# =============================================================================
# GPU-Accelerated Mask Functions
# =============================================================================

def create_circular_mask_gpu(
    shape: Tuple[int, int],
    radius: float,
    center: Tuple[float, float] | None = None,
    invert: bool = False,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """
    Create a circular binary mask on GPU.
    
    GPU-accelerated version of create_circular_mask using PyTorch.
    
    Args:
        shape: Output mask shape (height, width)
        radius: Radius of the circular region in pixels
        center: Center coordinates (y, x). If None, uses image center.
        invert: If True, mask is 0 inside circle, 1 outside.
        device: PyTorch device. If None, uses CUDA if available.
        
    Returns:
        Boolean tensor on specified device
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for GPU mask operations")
    
    h, w = shape
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if center is None:
        center = (h // 2, w // 2)
    
    cy, cx = center
    
    # Create coordinate grids on device
    y = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0)
    
    # Calculate distance from center
    dist_sq = (y - cy) ** 2 + (x - cx) ** 2
    
    # Create mask
    mask = dist_sq < radius ** 2
    
    if invert:
        mask = ~mask
    
    return mask


def dilate_mask_gpu(
    mask: "torch.Tensor",
    radius: int,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """
    Dilate a binary mask using max pooling on GPU.
    
    This is much faster than scipy.ndimage.binary_dilation for large masks.
    Uses max pooling with a circular kernel approximation.
    
    Args:
        mask: Input boolean tensor (H, W)
        radius: Dilation radius in pixels
        device: PyTorch device. If None, uses mask's device.
        
    Returns:
        Dilated mask tensor
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for GPU mask operations")
    
    if device is None:
        device = mask.device
    
    # Kernel size must be odd
    kernel_size = radius * 2 + 1
    padding = radius
    
    # Convert to float for max_pool2d (expects 4D: N, C, H, W)
    mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
    
    # Max pooling acts as dilation for binary masks
    dilated = torch.nn.functional.max_pool2d(
        mask_4d,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    
    # Convert back to boolean and remove batch/channel dims
    return dilated.squeeze(0).squeeze(0) > 0.5


def create_fft_mask_gpu(
    box_size: int,
    pixel_ang: float,
    inside_radius_ang: float,
    outside_radius_ang: float,
    threshold_mask: "torch.Tensor",
    expand_pixel: int = 10,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """
    Create the composite FFT mask for lattice spot removal on GPU.
    
    GPU-accelerated version of create_fft_mask. All operations stay on GPU
    to avoid CPU-GPU data transfers.
    
    Args:
        box_size: Size of the FFT (square)
        pixel_ang: Pixel size in Angstroms
        inside_radius_ang: Inner resolution limit (protect center)
        outside_radius_ang: Outer resolution limit (protect edges)
        threshold_mask: Boolean tensor from peak thresholding (True = peak)
        expand_pixel: Expansion radius for morphological dilation
        device: PyTorch device. If None, uses threshold_mask's device.
        
    Returns:
        Final mask tensor where True = keep, False = replace with inpainted values
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for GPU mask operations")
    
    if device is None:
        device = threshold_mask.device
    
    # Convert resolution to Fourier pixels
    inner_radius = resolution_to_pixels(inside_radius_ang, pixel_ang, box_size)
    outer_radius = resolution_to_pixels(outside_radius_ang, pixel_ang, box_size)
    
    # Clamp outer radius to valid range
    outer_radius = min(outer_radius, box_size // 2 - 1)
    
    # Create radial masks on GPU
    shape = (box_size, box_size)
    mask_center = create_circular_mask_gpu(shape, inner_radius, device=device)
    mask_outside = create_circular_mask_gpu(shape, outer_radius, device=device)
    
    # Ensure threshold_mask is boolean tensor on correct device
    if not isinstance(threshold_mask, torch.Tensor):
        threshold_mask = torch.from_numpy(threshold_mask).to(device)
    else:
        threshold_mask = threshold_mask.to(device)
    
    # Combine masks (same logic as CPU version)
    combined = ~threshold_mask | ~mask_outside | mask_center
    mask_final = combined
    
    # Expand the removal regions
    if expand_pixel > 0:
        removal_regions = ~mask_final
        
        rad_expand = expand_pixel // 2 - 1
        if rad_expand > 0:
            removal_regions = dilate_mask_gpu(removal_regions, rad_expand, device=device)
        
        mask_final = ~removal_regions
    
    return mask_final
