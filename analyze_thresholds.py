#!/usr/bin/env python
"""
Analyze threshold distribution across 100 test images.
Creates histogram for README documentation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lattice_subtraction import Config
from lattice_subtraction.threshold_optimizer import ThresholdOptimizer
from lattice_subtraction.io import read_mrc


def analyze_thresholds(image_dir: str, num_images: int = 100):
    """Analyze optimal thresholds for a set of images."""
    
    image_dir = Path(image_dir)
    mrc_files = sorted(image_dir.glob("*.mrc"))[:num_images]
    
    print(f"Analyzing {len(mrc_files)} images from {image_dir}")
    
    # Setup
    config = Config(pixel_ang=0.56)
    optimizer = ThresholdOptimizer(
        config,
        min_threshold=1.40,
        max_threshold=1.60,
    )
    
    thresholds = []
    peak_counts = []
    quality_scores = []
    
    for mrc_file in tqdm(mrc_files, desc="Analyzing"):
        try:
            image = read_mrc(mrc_file)
            result = optimizer.find_optimal(image)
            
            thresholds.append(result.threshold)
            peak_counts.append(result.peak_count)
            quality_scores.append(result.quality_score)
            
        except Exception as e:
            print(f"Error processing {mrc_file.name}: {e}")
            continue
    
    thresholds = np.array(thresholds)
    peak_counts = np.array(peak_counts)
    quality_scores = np.array(quality_scores)
    
    # Statistics
    print("\n" + "="*50)
    print("THRESHOLD DISTRIBUTION ANALYSIS")
    print("="*50)
    print(f"Images analyzed: {len(thresholds)}")
    print(f"\nThreshold Statistics:")
    print(f"  Mean:   {thresholds.mean():.3f}")
    print(f"  Std:    {thresholds.std():.3f}")
    print(f"  Min:    {thresholds.min():.3f}")
    print(f"  Max:    {thresholds.max():.3f}")
    print(f"  Median: {np.median(thresholds):.3f}")
    print(f"\nPeak Count Statistics:")
    print(f"  Mean:   {peak_counts.mean():.0f}")
    print(f"  Std:    {peak_counts.std():.0f}")
    print(f"  Range:  {peak_counts.min():.0f} - {peak_counts.max():.0f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Histogram of thresholds
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(thresholds, bins=21, range=(1.40, 1.60), 
                                      edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(thresholds.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {thresholds.mean():.3f}')
    ax1.axvline(1.42, color='orange', linestyle=':', linewidth=2, 
                label='v1.0 fixed: 1.42')
    ax1.set_xlabel('Optimal Threshold', fontsize=12)
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_title('Threshold Distribution (n=100)', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Threshold vs Peak Count scatter
    ax2 = axes[1]
    scatter = ax2.scatter(thresholds, peak_counts, c=quality_scores, 
                          cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Optimal Threshold', fontsize=12)
    ax2.set_ylabel('Peak Count', fontsize=12)
    ax2.set_title('Threshold vs Peak Count', fontsize=14)
    plt.colorbar(scatter, ax=ax2, label='Quality Score')
    ax2.grid(alpha=0.3)
    
    # Peak count histogram
    ax3 = axes[2]
    ax3.hist(peak_counts, bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
    ax3.axvline(600, color='red', linestyle='--', linewidth=2, label='Target: 600')
    ax3.axvline(peak_counts.mean(), color='orange', linestyle='--', linewidth=2, 
                label=f'Mean: {peak_counts.mean():.0f}')
    ax3.set_xlabel('Peak Count', fontsize=12)
    ax3.set_ylabel('Number of Images', fontsize=12)
    ax3.set_title('Peak Count Distribution', fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.suptitle('v1.1.0 Adaptive Threshold Optimization Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / "docs" / "images" / "threshold_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    # Also save to current directory
    plt.savefig("threshold_analysis.png", dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: threshold_analysis.png")
    
    plt.show()
    
    return thresholds, peak_counts, quality_scores


if __name__ == "__main__":
    # Default to test images
    image_dir = "/mnt/data_1/CU_Boulder/KASINATH/test_images/dose_weighted"
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    
    analyze_thresholds(image_dir, num_images=100)
