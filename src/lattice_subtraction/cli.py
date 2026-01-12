"""
Command-line interface for lattice subtraction.

This module provides a Click-based CLI for processing single files
or batch directories from the command line.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .config import Config, create_default_config
from .core import LatticeSubtractor
from .batch import BatchProcessor


# Setup logging
def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(version="1.0.0", prog_name="lattice-sub")
def main():
    """
    Lattice Subtraction for Cryo-EM Micrographs.
    
    Remove periodic crystal lattice signals from micrographs to reveal
    non-periodic features like defects, particles, or molecular tags.
    
    \b
    Examples:
        # Process single file
        lattice-sub process input.mrc -o output.mrc --pixel-size 0.56
        
        # Batch process directory
        lattice-sub batch input_dir/ output_dir/ --pixel-size 0.56 -j 8
        
        # Create default config file
        lattice-sub init-config params.yaml --pixel-size 0.56
    """
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False),
    help="Output file path. Default: sub_<input_name>",
)
@click.option(
    "--pixel-size", "-p",
    type=float,
    required=True,
    help="Pixel size in Angstroms",
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=1.42,
    help="Peak detection threshold. Default: 1.42",
)
@click.option(
    "--inside-radius",
    type=float,
    default=90.0,
    help="Inner resolution limit in Angstroms. Default: 90",
)
@click.option(
    "--outside-radius",
    type=float,
    default=None,
    help="Outer resolution limit in Angstroms. Default: auto",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to YAML config file (overrides other options)",
)
@click.option(
    "--gpu/--no-gpu",
    default=False,
    help="Use GPU acceleration (requires CuPy). Default: CPU",
)
@click.option(
    "--diagnostics/--no-diagnostics",
    default=False,
    help="Save diagnostic images (mask, power spectrum)",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def process(
    input_file: str,
    output: Optional[str],
    pixel_size: float,
    threshold: float,
    inside_radius: float,
    outside_radius: Optional[float],
    config: Optional[str],
    gpu: bool,
    diagnostics: bool,
    verbose: bool,
):
    """
    Process a single micrograph.
    
    INPUT_FILE: Path to input MRC file
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_file)
    
    # Determine output path
    if output is None:
        output_path = input_path.parent / f"sub_{input_path.name}"
    else:
        output_path = Path(output)
    
    # Load or create config
    if config:
        logger.info(f"Loading config from {config}")
        cfg = Config.from_yaml(config)
    else:
        cfg = Config(
            pixel_ang=pixel_size,
            threshold=threshold,
            inside_radius_ang=inside_radius,
            outside_radius_ang=outside_radius,
            backend="cupy" if gpu else "numpy",
        )
    
    logger.info(f"Processing: {input_path}")
    logger.info(f"Parameters: pixel={cfg.pixel_ang}Å, threshold={cfg.threshold}")
    
    # Process
    try:
        subtractor = LatticeSubtractor(cfg)
        result = subtractor.process(input_path, return_diagnostics=diagnostics)
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, pixel_size=cfg.pixel_ang)
        logger.info(f"Saved: {output_path}")
        
        # Save diagnostics if requested
        if diagnostics and result.fft_mask is not None:
            from .io import write_mrc
            import numpy as np
            
            mask_path = output_path.with_suffix(".mask.mrc")
            write_mrc(result.fft_mask.astype(np.float32), mask_path)
            logger.info(f"Saved mask: {mask_path}")
            
            if result.power_spectrum is not None:
                ps_path = output_path.with_suffix(".power.mrc")
                write_mrc(result.power_spectrum, ps_path)
                logger.info(f"Saved power spectrum: {ps_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
    
    logger.info("Done!")


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option(
    "--pixel-size", "-p",
    type=float,
    required=True,
    help="Pixel size in Angstroms",
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=1.42,
    help="Peak detection threshold. Default: 1.42",
)
@click.option(
    "--pattern",
    type=str,
    default="*.mrc",
    help="Glob pattern for input files. Default: *.mrc",
)
@click.option(
    "--prefix",
    type=str,
    default="sub_",
    help="Output filename prefix. Default: sub_",
)
@click.option(
    "-j", "--jobs",
    type=int,
    default=None,
    help="Number of parallel workers. Default: CPU count - 1",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to YAML config file",
)
@click.option(
    "-r", "--recursive",
    is_flag=True,
    help="Search subdirectories recursively",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def batch(
    input_dir: str,
    output_dir: str,
    pixel_size: float,
    threshold: float,
    pattern: str,
    prefix: str,
    jobs: Optional[int],
    config: Optional[str],
    recursive: bool,
    verbose: bool,
):
    """
    Batch process a directory of micrographs.
    
    \b
    INPUT_DIR: Directory containing input MRC files
    OUTPUT_DIR: Directory for processed output files
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Load or create config
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config(
            pixel_ang=pixel_size,
            threshold=threshold,
        )
    
    logger.info(f"Batch processing: {input_dir} -> {output_dir}")
    logger.info(f"Pattern: {pattern}, Workers: {jobs or 'auto'}")
    
    # Process
    processor = BatchProcessor(cfg, num_workers=jobs, output_prefix=prefix)
    result = processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
        recursive=recursive,
        show_progress=True,
    )
    
    # Report results
    logger.info(f"Completed: {result.successful}/{result.total} files ({result.success_rate:.1f}%)")
    
    if result.failed > 0:
        logger.warning(f"Failed files: {result.failed}")
        for path, error in result.failed_files[:10]:  # Show first 10
            logger.warning(f"  {path}: {error}")
        
        if len(result.failed_files) > 10:
            logger.warning(f"  ... and {len(result.failed_files) - 10} more")
        
        sys.exit(1)


@main.command("init-config")
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--pixel-size", "-p",
    type=float,
    default=0.56,
    help="Pixel size in Angstroms. Default: 0.56",
)
@click.option(
    "--detector",
    type=click.Choice(["K3", "Falcon", "generic"]),
    default="generic",
    help="Detector type for default settings",
)
def init_config(output_file: str, pixel_size: float, detector: str):
    """
    Create a default configuration file.
    
    OUTPUT_FILE: Path for the YAML config file
    """
    cfg = create_default_config(pixel_ang=pixel_size, detector=detector)
    cfg.to_yaml(output_file)
    
    click.echo(f"Created config file: {output_file}")
    click.echo(f"  Pixel size: {cfg.pixel_ang} Å")
    click.echo(f"  Threshold: {cfg.threshold}")
    click.echo(f"  Inside radius: {cfg.inside_radius_ang} Å")


@main.command("convert-config")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
def convert_config(input_file: str, output_file: str):
    """
    Convert legacy PARAMETER file to YAML format.
    
    \b
    INPUT_FILE: Path to legacy PARAMETER file
    OUTPUT_FILE: Path for new YAML config file
    """
    try:
        cfg = Config.from_legacy_parameter_file(input_file)
        cfg.to_yaml(output_file)
        click.echo(f"Converted {input_file} -> {output_file}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
