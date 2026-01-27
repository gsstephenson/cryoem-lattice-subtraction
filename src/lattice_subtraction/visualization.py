"""
Visualization utilities for lattice subtraction results.

This module provides functions to create comparison visualizations
showing original, processed, difference images, and threshold optimization curves.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

# Silence matplotlib's verbose debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Silence PIL/Pillow debug logging (PNG chunk messages)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def compute_threshold_curve(
    image: np.ndarray,
    config,
    n_points: int = 21,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute quality scores across a range of thresholds.
    
    Args:
        image: Original image array
        config: Config object with processing parameters
        n_points: Number of threshold points to evaluate
        
    Returns:
        Tuple of (thresholds, quality_scores, optimal_threshold, optimal_quality)
    """
    from .threshold_optimizer import ThresholdOptimizer
    
    optimizer = ThresholdOptimizer(config)
    
    # Prepare FFT data once
    subtracted, radial_mask, box_size = optimizer._prepare_fft_data(image)
    
    # Evaluate across threshold range
    thresholds = np.linspace(
        optimizer.min_threshold, 
        optimizer.max_threshold, 
        n_points
    )
    
    # Use GPU batch if available
    if optimizer.use_gpu:
        qualities, peak_counts = optimizer._compute_quality_batch_gpu(
            subtracted, radial_mask, thresholds
        )
    else:
        qualities = []
        for t in thresholds:
            q, _ = optimizer._compute_quality(subtracted, radial_mask, t)
            qualities.append(q)
        qualities = np.array(qualities)
    
    # Find optimal
    best_idx = np.argmax(qualities)
    optimal_threshold = thresholds[best_idx]
    optimal_quality = qualities[best_idx]
    
    return thresholds, qualities, optimal_threshold, optimal_quality


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


def create_comparison_figure_with_threshold(
    original: np.ndarray,
    processed: np.ndarray,
    thresholds: np.ndarray,
    quality_scores: np.ndarray,
    optimal_threshold: float,
    optimal_quality: float,
    title: str = "Lattice Subtraction Comparison",
    figsize: Tuple[int, int] = (24, 6),
    dpi: int = 150,
):
    """
    Create a 4-panel comparison figure with threshold optimization curve.
    
    Layout: [Original] [Subtracted] [Difference] [Threshold vs Quality]
    
    Args:
        original: Original image array
        processed: Processed (lattice-subtracted) image array
        thresholds: Array of threshold values tested
        quality_scores: Array of quality scores for each threshold
        optimal_threshold: The optimal threshold that was selected
        optimal_quality: Quality score at optimal threshold
        title: Figure title
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saving
        
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    
    # Compute difference
    difference = original - processed
    
    # Create figure with 4 panels (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Contrast limits from original
    vmin, vmax = np.percentile(original, [1, 99])
    
    # Panel 1: Original
    axes[0].imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Original\n{original.shape}')
    axes[0].axis('off')
    
    # Panel 2: Lattice Subtracted
    axes[1].imshow(processed, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Lattice Subtracted\n{processed.shape}')
    axes[1].axis('off')
    
    # Panel 3: Difference (removed lattice)
    diff_std = np.std(difference)
    axes[2].imshow(
        difference, 
        cmap='RdBu_r',
        vmin=-diff_std * 3, 
        vmax=diff_std * 3
    )
    axes[2].set_title('Difference (Removed Lattice)')
    axes[2].axis('off')
    
    # Panel 4: Threshold vs Quality Score curve
    axes[3].plot(thresholds, quality_scores, 'b-', linewidth=2, label='Quality Score')
    axes[3].axvline(x=optimal_threshold, color='r', linestyle='--', linewidth=2, 
                    label=f'Optimal: {optimal_threshold:.3f}')
    axes[3].scatter([optimal_threshold], [optimal_quality], color='r', s=100, zorder=5)
    axes[3].set_xlabel('Threshold', fontsize=11)
    axes[3].set_ylabel('Lattice Removal Efficacy', fontsize=11)
    axes[3].set_title(f'Threshold Optimization\nOptimal = {optimal_threshold:.3f}')
    axes[3].legend(loc='best', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(thresholds.min(), thresholds.max())
    
    # Title
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def save_comparison_visualization(
    original_path: Path,
    processed_path: Path,
    output_path: Path,
    config = None,
    dpi: int = 150,
) -> None:
    """
    Create and save a 4-panel comparison visualization for a single image pair.
    
    Includes threshold optimization curve showing how the optimal threshold
    was selected based on lattice removal efficacy.
    
    Args:
        original_path: Path to original MRC file
        processed_path: Path to processed MRC file
        output_path: Path for output PNG file
        config: Config object for threshold computation (optional, will use defaults)
        dpi: Resolution for output images
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
    short_name = name[:50] + "..." if len(name) > 50 else name
    title = f"Lattice Subtraction: {short_name}"
    
    # Try to compute threshold curve if config available
    try:
        if config is None:
            # Create default config for threshold computation
            from .config import Config
            config = Config(pixel_ang=0.56)  # Default K3 pixel size
        
        # Compute threshold optimization curve
        thresholds, quality_scores, optimal_threshold, optimal_quality = \
            compute_threshold_curve(original, config)
        
        # Create 4-panel figure with threshold curve
        fig = create_comparison_figure_with_threshold(
            original, processed,
            thresholds, quality_scores,
            optimal_threshold, optimal_quality,
            title=title, dpi=dpi
        )
    except Exception as e:
        # Fallback to 3-panel if threshold computation fails
        logger.debug(f"Could not compute threshold curve: {e}, using 3-panel view")
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
    limit: Optional[int] = None,
    config = None,
) -> Tuple[int, int]:
    """
    Generate comparison visualizations for processed images in a directory.
    
    Args:
        input_dir: Directory containing original MRC files
        output_dir: Directory containing processed MRC files
        viz_dir: Directory for output visualization PNG files
        prefix: Prefix used for processed files (default: "sub_")
        pattern: Glob pattern for finding processed files
        dpi: Resolution for output images
        show_progress: Show progress bar
        limit: Maximum number of visualizations to generate (None = all)
        config: Config object for threshold computation (optional)
        
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
    
    # Apply limit if specified
    total_available = len(output_files)
    if limit is not None and limit > 0:
        output_files = output_files[:limit]
        logger.info(f"Limiting to {limit} visualizations (of {total_available} available)")
    
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
            
            # Generate visualization with 4-panel layout
            save_comparison_visualization(
                original_path=input_path,
                processed_path=processed_path,
                output_path=viz_path,
                config=config,
                dpi=dpi,
            )
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to create visualization for {processed_path.name}: {e}")
    
    return successful, total
