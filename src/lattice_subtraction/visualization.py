"""
Visualization utilities for lattice subtraction results.

This module provides functions to create comparison visualizations
showing original, processed, and difference images.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def create_comparison_figure(
    original: np.ndarray,
    processed: np.ndarray,
    title: str = "Lattice Subtraction Comparison",
    figsize: Tuple[int, int] = (18, 6),
    dpi: int = 150,
):
    """
    Create a comparison figure showing original, processed, and difference images.
    
    Args:
        original: Original image array
        processed: Processed (lattice-subtracted) image array
        title: Figure title
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saving
        
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    
    # Compute difference
    difference = original - processed
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Contrast limits from original
    vmin, vmax = np.percentile(original, [1, 99])
    
    # Original
    axes[0].imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Original\n{original.shape}')
    axes[0].axis('off')
    
    # Lattice Subtracted
    axes[1].imshow(processed, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Lattice Subtracted\n{processed.shape}')
    axes[1].axis('off')
    
    # Difference (removed lattice)
    diff_std = np.std(difference)
    axes[2].imshow(
        difference, 
        cmap='RdBu_r',
        vmin=-diff_std * 3, 
        vmax=diff_std * 3
    )
    axes[2].set_title('Difference (Removed Lattice)')
    axes[2].axis('off')
    
    # Title
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def save_comparison_visualization(
    original_path: Path,
    processed_path: Path,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Create and save a comparison visualization for a single image pair.
    
    Args:
        original_path: Path to original MRC file
        processed_path: Path to processed MRC file
        output_path: Path for output PNG file
    """
    import matplotlib.pyplot as plt
    import mrcfile
    
    # Load images
    with mrcfile.open(original_path, 'r') as f:
        original = f.data.copy()
    with mrcfile.open(processed_path, 'r') as f:
        processed = f.data.copy()
    
    # Create title
    name = original_path.name
    short_name = name[:60] + "..." if len(name) > 60 else name
    title = f"Lattice Subtraction Comparison: {short_name}"
    
    # Create and save figure
    fig = create_comparison_figure(original, processed, title=title, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_visualizations(
    input_dir: Path,
    output_dir: Path,
    viz_dir: Path,
    prefix: str = "sub_",
    pattern: str = "*.mrc",
    dpi: int = 150,
    show_progress: bool = True,
) -> Tuple[int, int]:
    """
    Generate comparison visualizations for all processed images in a directory.
    
    Args:
        input_dir: Directory containing original MRC files
        output_dir: Directory containing processed MRC files
        viz_dir: Directory for output visualization PNG files
        prefix: Prefix used for processed files (default: "sub_")
        pattern: Glob pattern for finding processed files
        dpi: Resolution for output images
        show_progress: Show progress bar
        
    Returns:
        Tuple of (successful_count, total_count)
    """
    import matplotlib.pyplot as plt
    
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all processed files
    output_files = sorted(Path(output_dir).glob(f"{prefix}{pattern}"))
    
    if not output_files:
        logger.warning(f"No processed files found matching '{prefix}{pattern}' in {output_dir}")
        return 0, 0
    
    successful = 0
    total = len(output_files)
    
    # Setup iterator with optional progress bar
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(output_files, desc="Generating visualizations", unit="file")
        except ImportError:
            iterator = output_files
    else:
        iterator = output_files
    
    for processed_path in iterator:
        try:
            # Get corresponding input file
            input_name = processed_path.name.replace(prefix, "", 1)
            input_path = Path(input_dir) / input_name
            
            if not input_path.exists():
                logger.debug(f"Original not found: {input_path}")
                continue
            
            # Output path
            viz_name = input_name.replace(".mrc", ".png")
            viz_path = viz_dir / viz_name
            
            # Skip if already exists
            if viz_path.exists():
                successful += 1
                continue
            
            # Generate visualization
            save_comparison_visualization(
                original_path=input_path,
                processed_path=processed_path,
                output_path=viz_path,
                dpi=dpi,
            )
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to create visualization for {processed_path.name}: {e}")
    
    return successful, total
