#!/usr/bin/env python3
"""
Simulate motion correction by copying files from input to input_live at random intervals.

This script mimics the behavior of a motion correction pipeline that produces
output files over time, allowing testing of the live watch mode.
"""

import argparse
import shutil
import time
import random
from pathlib import Path


def simulate_motion_correction(
    source_dir: Path,
    dest_dir: Path,
    min_delay: float = 2.0,
    max_delay: float = 10.0,
    num_files: int = None,
):
    """
    Copy files from source to destination with random delays.
    
    Args:
        source_dir: Source directory with MRC files
        dest_dir: Destination directory (input_live)
        min_delay: Minimum delay between copies (seconds)
        max_delay: Maximum delay between copies (seconds)
        num_files: Number of files to copy (None = all)
    """
    # Get list of MRC files
    mrc_files = sorted(source_dir.glob("*.mrc"))
    
    if not mrc_files:
        print(f"Error: No MRC files found in {source_dir}")
        return
    
    if num_files:
        mrc_files = mrc_files[:num_files]
    
    print(f"Will copy {len(mrc_files)} files from {source_dir} to {dest_dir}")
    print(f"Delay range: {min_delay}-{max_delay} seconds")
    print(f"Press Ctrl+C to stop\n")
    
    # Ensure destination is empty
    for existing_file in dest_dir.glob("*.mrc"):
        print(f"Removing existing file: {existing_file.name}")
        existing_file.unlink()
    
    try:
        for i, src_file in enumerate(mrc_files, 1):
            # Random delay to simulate processing time
            delay = random.uniform(min_delay, max_delay)
            
            print(f"[{i}/{len(mrc_files)}] Waiting {delay:.1f}s before copying {src_file.name}...")
            time.sleep(delay)
            
            # Copy file
            dest_file = dest_dir / src_file.name
            shutil.copy2(src_file, dest_file)
            print(f"[{i}/{len(mrc_files)}] ✓ Copied {src_file.name}")
        
        print(f"\n✓ Finished copying all {len(mrc_files)} files")
    
    except KeyboardInterrupt:
        print(f"\n\nStopped by user. Copied {i-1}/{len(mrc_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate motion correction output by copying files at random intervals"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Source directory containing MRC files"
    )
    parser.add_argument(
        "dest_dir",
        type=Path,
        help="Destination directory (will be cleared first)"
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=2.0,
        help="Minimum delay between files in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=10.0,
        help="Maximum delay between files in seconds (default: 10.0)"
    )
    parser.add_argument(
        "-n", "--num-files",
        type=int,
        default=None,
        help="Number of files to copy (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.source_dir.exists():
        print(f"Error: Source directory does not exist: {args.source_dir}")
        return 1
    
    if not args.dest_dir.exists():
        args.dest_dir.mkdir(parents=True, exist_ok=True)
    
    simulate_motion_correction(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        num_files=args.num_files,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
