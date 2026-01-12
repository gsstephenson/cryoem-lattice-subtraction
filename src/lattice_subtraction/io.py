"""
MRC file I/O utilities.

This module wraps the mrcfile library for reading and writing MRC format files
commonly used in cryo-EM.
"""

from pathlib import Path
from typing import Optional
import numpy as np

try:
    import mrcfile
except ImportError:
    raise ImportError(
        "mrcfile is required for MRC I/O. Install with: pip install mrcfile"
    )


def read_mrc(
    path: str | Path,
    as_float32: bool = True,
) -> np.ndarray:
    """
    Read a 2D micrograph from an MRC file.
    
    Args:
        path: Path to MRC file
        as_float32: If True, convert to float32. Default: True
        
    Returns:
        2D numpy array containing the image data
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file contains 3D data (use read_mrc_stack instead)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRC file not found: {path}")
    
    with mrcfile.open(path, mode='r', permissive=True) as mrc:
        data = mrc.data.copy()
    
    # Handle 3D MRC files (single slice)
    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        else:
            raise ValueError(
                f"Expected 2D micrograph, got 3D stack with shape {data.shape}. "
                "Use read_mrc_stack() for 3D data."
            )
    
    if as_float32:
        data = data.astype(np.float32)
    
    return data


def read_mrc_stack(
    path: str | Path,
    as_float32: bool = True,
) -> np.ndarray:
    """
    Read a 3D stack from an MRC file.
    
    Args:
        path: Path to MRC file
        as_float32: If True, convert to float32. Default: True
        
    Returns:
        3D numpy array with shape (nz, ny, nx)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRC file not found: {path}")
    
    with mrcfile.open(path, mode='r', permissive=True) as mrc:
        data = mrc.data.copy()
    
    if as_float32:
        data = data.astype(np.float32)
    
    return data


def read_mrc_header(path: str | Path) -> dict:
    """
    Read only the header information from an MRC file.
    
    Args:
        path: Path to MRC file
        
    Returns:
        Dictionary containing header information including:
        - shape: (nx, ny, nz)
        - pixel_size: voxel size in Angstroms
        - mode: data type mode
        - statistics: (min, max, mean, rms)
    """
    path = Path(path)
    
    with mrcfile.open(path, mode='r', permissive=True) as mrc:
        header = {
            'shape': (int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)),
            'pixel_size': (
                float(mrc.voxel_size.x),
                float(mrc.voxel_size.y), 
                float(mrc.voxel_size.z)
            ),
            'mode': int(mrc.header.mode),
            'statistics': (
                float(mrc.header.dmin),
                float(mrc.header.dmax),
                float(mrc.header.dmean),
                float(mrc.header.rms),
            ),
        }
    
    return header


def write_mrc(
    data: np.ndarray,
    path: str | Path,
    pixel_size: float = 1.0,
    overwrite: bool = True,
) -> None:
    """
    Write a 2D or 3D array to an MRC file.
    
    Args:
        data: 2D or 3D numpy array to write
        path: Output file path
        pixel_size: Pixel/voxel size in Angstroms. Default: 1.0
        overwrite: If True, overwrite existing file. Default: True
        
    Raises:
        FileExistsError: If file exists and overwrite=False
        ValueError: If data has invalid shape
    """
    path = Path(path)
    
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to float32 for compatibility
    if data.dtype not in (np.float32, np.int16, np.uint16, np.int8, np.uint8):
        data = data.astype(np.float32)
    
    # Ensure contiguous array
    data = np.ascontiguousarray(data)
    
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = pixel_size
        
        # Update statistics
        mrc.update_header_stats()


def get_pixel_size_from_mrc(path: str | Path) -> float:
    """
    Extract pixel size from MRC file header.
    
    Args:
        path: Path to MRC file
        
    Returns:
        Pixel size in Angstroms (from X dimension)
    """
    header = read_mrc_header(path)
    return header['pixel_size'][0]
