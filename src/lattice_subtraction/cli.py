"""
Command-line interface for lattice subtraction.

This module provides a Click-based CLI for processing single files
or batch directories from the command line.

Terminal output is styled when running interactively. When piped
or used in a pipeline, decorative output is automatically suppressed.
"""

import logging
import sys
from os import cpu_count as import_cpu_count
from pathlib import Path
from typing import Optional

import click

from .config import Config, create_default_config
from .core import LatticeSubtractor
from .batch import BatchProcessor
from .visualization import generate_visualizations, save_comparison_visualization
from .ui import get_ui, get_gpu_name
from .io import read_mrc


# Setup logging - minimal format when interactive UI is active
def setup_logging(verbose: bool, interactive: bool = False) -> None:
    """Configure logging based on verbosity and interactivity."""
    if interactive:
        # Suppress logging when using interactive UI
        level = logging.DEBUG if verbose else logging.WARNING
    else:
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
    
    When running interactively, displays a styled terminal interface
    with ASCII banner and colored progress output. Use --quiet to
    suppress decorative output, or pipe output to automatically
    disable the interactive interface.
    
    \b
    Examples:
        # Process single file with GPU
        lattice-sub process input.mrc -o output.mrc --pixel-size 0.56 --gpu
        
        # Batch process directory (8 parallel workers)
        lattice-sub batch input_dir/ output_dir/ --pixel-size 0.56 -j 8
        
        # Batch with visualizations
        lattice-sub batch input_dir/ output_dir/ -p 0.56 --vis viz_dir/
        
        # Generate visualizations for existing files
        lattice-sub visualize input_dir/ output_dir/ viz_dir/
        
        # Create config file
        lattice-sub init-config params.yaml --pixel-size 0.56
        
        # Quiet mode (no banner/colors)
        lattice-sub process input.mrc -o output.mrc -p 0.56 --quiet
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
    help="Use GPU acceleration (requires PyTorch+CUDA). Default: CPU",
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
@click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Suppress decorative output (banner, colors)",
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
    quiet: bool,
):
    """
    Process a single micrograph.
    
    INPUT_FILE: Path to input MRC file
    """
    # Initialize UI
    ui = get_ui(quiet=quiet)
    ui.print_banner()
    
    setup_logging(verbose, interactive=ui.interactive)
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
            backend="pytorch" if gpu else "numpy",
        )
    
    # Print configuration
    gpu_name = get_gpu_name() if gpu else None
    ui.print_config(cfg.pixel_ang, cfg.threshold, cfg.backend, gpu_name)
    ui.start_timer()
    
    logger.info(f"Processing: {input_path}")
    logger.info(f"Parameters: pixel={cfg.pixel_ang}Å, threshold={cfg.threshold}")
    
    # Process
    try:
        # Get image shape for display
        image = read_mrc(input_path)
        ui.start_processing(input_path.name, shape=image.shape)
        
        subtractor = LatticeSubtractor(cfg)
        result = subtractor.process(image, return_diagnostics=diagnostics)
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, pixel_size=cfg.pixel_ang)
        ui.print_saved(str(output_path))
        logger.info(f"Saved: {output_path}")
        
        # Save diagnostics if requested
        if diagnostics and result.fft_mask is not None:
            from .io import write_mrc
            import numpy as np
            
            mask_path = output_path.with_suffix(".mask.mrc")
            write_mrc(result.fft_mask.astype(np.float32), mask_path)
            ui.print_saved(str(mask_path))
            logger.info(f"Saved mask: {mask_path}")
            
            if result.power_spectrum is not None:
                ps_path = output_path.with_suffix(".power.mrc")
                write_mrc(result.power_spectrum, ps_path)
                ui.print_saved(str(ps_path))
                logger.info(f"Saved power spectrum: {ps_path}")
        
        ui.end_processing(str(output_path), success=True)
        ui.print_summary(processed=1)
        
    except Exception as e:
        ui.end_processing(str(output_path), success=False)
        ui.print_error(str(e))
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


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
    "--vis",
    type=click.Path(file_okay=False),
    default=None,
    help="Generate comparison visualizations in this directory",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Suppress decorative output (banner, colors)",
)
@click.option(
    "--gpu/--no-gpu",
    default=False,
    help="Use GPU acceleration (requires PyTorch+CUDA). Default: CPU",
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
    vis: Optional[str],
    verbose: bool,
    quiet: bool,
    gpu: bool,
):
    """
    Batch process a directory of micrographs.
    
    \b
    INPUT_DIR: Directory containing input MRC files
    OUTPUT_DIR: Directory for processed output files
    """
    # Initialize UI
    ui = get_ui(quiet=quiet)
    ui.print_banner()
    
    setup_logging(verbose, interactive=ui.interactive)
    logger = logging.getLogger(__name__)
    
    # Load or create config
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config(
            pixel_ang=pixel_size,
            threshold=threshold,
            backend="pytorch" if gpu else "numpy",
        )
    
    # Count files first
    input_path = Path(input_dir)
    if recursive:
        files = list(input_path.rglob(pattern))
    else:
        files = list(input_path.glob(pattern))
    
    num_files = len(files)
    num_workers = jobs or max(1, (import_cpu_count() or 4) - 1)
    
    # Print configuration
    gpu_name = get_gpu_name() if gpu else None
    ui.print_config(cfg.pixel_ang, cfg.threshold, cfg.backend, gpu_name)
    ui.print_batch_header(num_files, output_dir, num_workers)
    ui.start_timer()
    
    logger.info(f"Batch processing: {input_dir} -> {output_dir}")
    logger.info(f"Pattern: {pattern}, Workers: {jobs or 'auto'}")
    
    # Process
    processor = BatchProcessor(cfg, num_workers=jobs, output_prefix=prefix)
    result = processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
        recursive=recursive,
        show_progress=True,  # Always show progress bar
    )
    
    # Report results with UI
    ui.print_batch_complete()
    ui.print_summary(processed=result.successful, failed=result.failed)
    
    logger.info(f"Completed: {result.successful}/{result.total} files ({result.success_rate:.1f}%)")
    
    if result.failed > 0:
        for path, error in result.failed_files[:5]:
            ui.print_error(f"{path}: {error}")
        if len(result.failed_files) > 5:
            ui.print_error(f"... and {len(result.failed_files) - 5} more failures")
        
        logger.warning(f"Failed files: {result.failed}")
        for path, error in result.failed_files[:10]:
            logger.warning(f"  {path}: {error}")
        
        if len(result.failed_files) > 10:
            logger.warning(f"  ... and {len(result.failed_files) - 10} more")
        
        sys.exit(1)
    
    # Generate visualizations if requested
    if vis:
        logger.info(f"Generating visualizations in: {vis}")
        viz_success, viz_total = generate_visualizations(
            input_dir=input_dir,
            output_dir=output_dir,
            viz_dir=vis,
            prefix=prefix,
            pattern=pattern,
            show_progress=True,
        )
        logger.info(f"Visualizations: {viz_success}/{viz_total} created")


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


@main.command("visualize")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("viz_dir", type=click.Path(file_okay=False))
@click.option(
    "--prefix",
    type=str,
    default="sub_",
    help="Prefix used for processed files. Default: sub_",
)
@click.option(
    "--pattern",
    type=str,
    default="*.mrc",
    help="Glob pattern for MRC files. Default: *.mrc",
)
@click.option(
    "--dpi",
    type=int,
    default=150,
    help="Resolution for output images. Default: 150",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def visualize(
    input_dir: str,
    output_dir: str,
    viz_dir: str,
    prefix: str,
    pattern: str,
    dpi: int,
    verbose: bool,
):
    """
    Generate comparison visualizations for processed micrographs.
    
    Creates side-by-side PNG images showing original, lattice-subtracted,
    and difference images for each processed micrograph.
    
    \b
    INPUT_DIR: Directory containing original MRC files
    OUTPUT_DIR: Directory containing processed (sub_*) MRC files
    VIZ_DIR: Directory for output visualization PNG files
    
    \b
    Example:
        lattice-sub visualize raw_images/ processed/ visualizations/
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating visualizations...")
    logger.info(f"  Original images: {input_dir}")
    logger.info(f"  Processed images: {output_dir}")
    logger.info(f"  Output visualizations: {viz_dir}")
    
    successful, total = generate_visualizations(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        viz_dir=Path(viz_dir),
        prefix=prefix,
        pattern=pattern,
        dpi=dpi,
        show_progress=True,
    )
    
    logger.info(f"Completed: {successful}/{total} visualizations created")
    
    if successful < total:
        logger.warning(f"Some visualizations failed: {total - successful}")
        sys.exit(1)


if __name__ == "__main__":
    main()
