"""
Command-line interface for lattice subtraction.

This module provides a Click-based CLI for processing single files
or batch directories from the command line.

Terminal output is styled when running interactively. When piped
or used in a pipeline, decorative output is automatically suppressed.
"""

import logging
import re
import subprocess
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
from .watch import LiveBatchProcessor


# CUDA version to PyTorch index URL mapping
# Note: PyTorch 2.5+ bundles CUDA 12.x by default, so explicit CUDA wheels
# are often not needed. This mapping is for cases where reinstallation helps.
CUDA_INDEX_URLS = {
    "11.8": "https://download.pytorch.org/whl/cu118",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "12.4": "https://download.pytorch.org/whl/cu124",  # Tested
    "12.6": "https://download.pytorch.org/whl/cu126",
    "12.8": "https://download.pytorch.org/whl/cu128",
}

# CUDA versions that can use newer wheel versions (backward compatible)
CUDA_FALLBACK = {
    "13.0": "12.8",
    "13.1": "12.8",
    "13.2": "12.8",
}

RECOMMENDED_CUDA = "12.8"


def detect_cuda_version() -> Optional[str]:
    """Detect CUDA version from nvidia-smi output.
    
    Returns:
        CUDA version string (e.g., "12.4") or None if not available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        
        # Get CUDA version from nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Parse "CUDA Version: 12.4" from output
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def get_pytorch_index_url(cuda_version: str) -> Optional[str]:
    """Get PyTorch index URL for a CUDA version.
    
    Args:
        cuda_version: CUDA version string (e.g., "12.4")
        
    Returns:
        PyTorch index URL or None if version not supported.
    """
    # Try exact match first
    if cuda_version in CUDA_INDEX_URLS:
        return CUDA_INDEX_URLS[cuda_version]
    
    # Check fallback for newer CUDA versions (backward compatible)
    if cuda_version in CUDA_FALLBACK:
        fallback = CUDA_FALLBACK[cuda_version]
        return CUDA_INDEX_URLS.get(fallback)
    
    # Try major.minor prefix match for minor version differences
    major_minor = ".".join(cuda_version.split(".")[:2])
    for version, url in CUDA_INDEX_URLS.items():
        if major_minor == ".".join(version.split(".")[:2]):
            return url
    
    return None


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


from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="lattice-sub")
def main():
    """
    Lattice Subtraction for Cryo-EM Micrographs.
    
    Remove periodic crystal lattice signals from micrographs to reveal
    non-periodic features like defects, particles, or molecular tags.
    
    GPU acceleration works automatically with PyTorch 2.5+. No setup needed!
    Use 'lattice-sub setup-gpu' to verify GPU status or troubleshoot.
    
    \b
    Quick Start:
        pip install lattice-sub
        lattice-sub process input.mrc -o output.mrc -p 0.56
    
    \b
    Examples:
        # Process single file (auto GPU detection)
        lattice-sub process input.mrc -o output.mrc --pixel-size 0.56
        
        # Force CPU processing
        lattice-sub process input.mrc -o output.mrc -p 0.56 --cpu
        
        # Batch process directory (GPU handles parallelism)
        lattice-sub batch input_dir/ output_dir/ --pixel-size 0.56
        
        # Batch with visualizations (4-panel with threshold curve)
        lattice-sub batch input_dir/ output_dir/ -p 0.56 --vis viz_dir/
        
        # Batch with limited visualizations (only first 10)
        lattice-sub batch input_dir/ output_dir/ -p 0.56 --vis viz_dir/ -n 10
        
        # CPU batch with parallel workers (use -j only with --cpu)
        lattice-sub batch input_dir/ output_dir/ -p 0.56 --cpu -j 8
        
        # Generate visualizations for existing files
        lattice-sub visualize input_dir/ output_dir/ viz_dir/
        
        # Create config file
        lattice-sub init-config params.yaml --pixel-size 0.56
        
        # Quiet mode (no banner/colors)
        lattice-sub process input.mrc -o output.mrc -p 0.56 --quiet
    """
    pass


@main.command("setup-gpu")
@click.option(
    "-y", "--yes",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force reinstall even if GPU is already working",
)
def setup_gpu(yes: bool, force: bool):
    """
    One-time GPU setup - installs PyTorch with CUDA support.
    
    Detects your CUDA version and installs the appropriate PyTorch
    wheels for GPU acceleration. You only need to run this once.
    
    Note: PyTorch 2.5+ often bundles CUDA support by default. This
    command will first check if your GPU is already working.
    
    \b
    Example:
        lattice-sub setup-gpu       # Interactive
        lattice-sub setup-gpu -y    # Skip confirmation
        lattice-sub setup-gpu --force  # Reinstall even if working
    """
    # First, check if GPU is already working
    click.echo("\nChecking current GPU status...")
    try:
        import torch
        if torch.cuda.is_available() and not force:
            gpu_name = torch.cuda.get_device_name(0)
            click.echo(f"\n✓ GPU already enabled: {gpu_name}")
            click.echo(f"  PyTorch version: {torch.__version__}")
            click.echo("\n  No setup needed! Your GPU is ready to use.")
            click.echo("  Use --force to reinstall anyway.")
            sys.exit(0)
        elif torch.cuda.is_available() and force:
            gpu_name = torch.cuda.get_device_name(0)
            click.echo(f"\n  GPU currently working: {gpu_name}")
            click.echo("  Proceeding with reinstall due to --force...")
    except ImportError:
        click.echo("  PyTorch not installed, proceeding with setup...")
    except Exception as e:
        click.echo(f"  Could not check GPU: {e}")
    
    click.echo("\nDetecting CUDA version...", nl=False)
    
    cuda_version = detect_cuda_version()
    
    if cuda_version is None:
        click.echo(" not found")
        click.echo("\n✗ No NVIDIA GPU detected.")
        click.echo("  Make sure nvidia-smi works and NVIDIA drivers are installed.")
        click.echo("  The package will run on CPU without GPU setup.")
        sys.exit(1)
    
    click.echo(f" found CUDA {cuda_version}")
    
    # Check for fallback version
    effective_version = CUDA_FALLBACK.get(cuda_version, cuda_version)
    if effective_version != cuda_version:
        click.echo(f"  (Using CUDA {effective_version} wheels - backward compatible)")
    
    index_url = get_pytorch_index_url(cuda_version)
    if index_url is None:
        supported = ", ".join(sorted(CUDA_INDEX_URLS.keys()))
        click.echo(f"\n✗ CUDA {cuda_version} is not supported.")
        click.echo(f"  Supported versions: {supported}")
        click.echo("\n  However, your GPU may already work with the bundled CUDA.")
        click.echo("  Try running: python -c \"import torch; print(torch.cuda.is_available())\"")
        sys.exit(1)
    
    # Build pip command
    pip_cmd = f"pip install torch --index-url {index_url} --force-reinstall"
    
    # Show what will happen
    click.echo(f"\nThis will install PyTorch with GPU support:")
    click.echo(f"  {pip_cmd}")
    
    if effective_version in ["12.4", "12.6", "12.8"]:
        click.echo(f"\n✓ CUDA {cuda_version} is well supported")
    else:
        click.echo(f"\nNote: CUDA 12.x is recommended, but {cuda_version} should work")
    
    click.echo("\nThis is a one-time setup. You won't need to run this again.")
    
    # Confirm unless --yes
    if not yes:
        if not click.confirm("\nProceed?", default=True):
            click.echo("Cancelled.")
            sys.exit(0)
    
    # Run pip install
    click.echo("\nInstalling PyTorch with CUDA support...")
    
    try:
        result = subprocess.run(
            pip_cmd.split(),
            check=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"\n✗ Installation failed with exit code {e.returncode}")
        click.echo("  Try running the pip command manually:")
        click.echo(f"  {pip_cmd}")
        sys.exit(1)
    
    # Verify installation
    click.echo("\nVerifying GPU access...")
    try:
        import importlib
        import torch
        importlib.reload(torch)  # Reload in case it was imported before
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            click.echo(f"\n✓ GPU enabled: {gpu_name}")
            click.echo("\nYou can now use lattice-sub with automatic GPU acceleration!")
        else:
            click.echo("\n⚠ PyTorch installed but CUDA not available.")
            click.echo("  This may require a Python restart. Try:")
            click.echo("  python -c \"import torch; print(torch.cuda.is_available())\"")
    except ImportError:
        click.echo("\n⚠ Could not verify - restart Python and check manually:")
        click.echo("  python -c \"import torch; print(torch.cuda.is_available())\"")


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
    default=None,
    help="Peak detection threshold. Default: auto (GPU-optimized per-image)",
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
    "--cpu",
    is_flag=True,
    help="Force CPU processing (disable GPU auto-detection)",
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
    threshold: Optional[float],
    inside_radius: float,
    outside_radius: Optional[float],
    config: Optional[str],
    cpu: bool,
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
        # Use "auto" threshold if not specified (GPU-optimized per-image)
        thresh_value = threshold if threshold is not None else "auto"
        cfg = Config(
            pixel_ang=pixel_size,
            threshold=thresh_value,
            inside_radius_ang=inside_radius,
            outside_radius_ang=outside_radius,
            backend="numpy" if cpu else "auto",
        )
    
    # Print configuration
    gpu_name = get_gpu_name() if not cpu else None
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
    default=None,
    help="Peak detection threshold. Default: auto (GPU-optimized per-image)",
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
    help="Number of parallel workers. Default: 1 for GPU, CPU count - 1 for --cpu mode",
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
    "-n", "--num-vis",
    type=int,
    default=None,
    help="Number of visualizations to generate (default: all)",
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
    "--cpu",
    is_flag=True,
    help="Force CPU processing (disable GPU auto-detection)",
)
@click.option(
    "--live",
    is_flag=True,
    help="Watch mode: continuously monitor input directory for new files (Press Ctrl+C to stop)",
)
def batch(
    input_dir: str,
    output_dir: str,
    pixel_size: float,
    threshold: Optional[float],
    pattern: str,
    prefix: str,
    jobs: Optional[int],
    config: Optional[str],
    recursive: bool,
    vis: Optional[str],
    num_vis: Optional[int],
    verbose: bool,
    quiet: bool,
    cpu: bool,
    live: bool,
):
    """
    Batch process a directory of micrographs.
    
    \b
    INPUT_DIR: Directory containing input MRC files
    OUTPUT_DIR: Directory for processed output files
    """
    # Validate options
    if live and recursive:
        click.echo("Error: --live and --recursive cannot be used together", err=True)
        sys.exit(1)
    
    # Initialize UI
    ui = get_ui(quiet=quiet)
    ui.print_banner()
    
    setup_logging(verbose, interactive=ui.interactive)
    logger = logging.getLogger(__name__)
    
    # Load or create config
    if config:
        cfg = Config.from_yaml(config)
    else:
        # Use "auto" threshold if not specified (GPU-optimized per-image)
        thresh_value = threshold if threshold is not None else "auto"
        cfg = Config(
            pixel_ang=pixel_size,
            threshold=thresh_value,
            backend="numpy" if cpu else "auto",
        )
    
    # Print configuration
    gpu_name = get_gpu_name() if not cpu else None
    ui.print_config(cfg.pixel_ang, cfg.threshold, cfg.backend, gpu_name)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # LIVE WATCH MODE
    if live:
        logger.info(f"Starting live watch mode: {input_dir} -> {output_dir}")
        
        # Determine number of workers
        if jobs is not None:
            num_workers = jobs
        elif cpu:
            num_workers = max(1, (import_cpu_count() or 4) - 1)
        else:
            # For GPU: use 1 worker per GPU (or 1 if single GPU)
            try:
                import torch
                num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
            except ImportError:
                num_workers = 1
        
        ui.show_watch_startup(str(input_path))
        ui.start_timer()
        
        # Create live processor
        live_processor = LiveBatchProcessor(
            config=cfg,
            output_prefix=prefix,
            debounce_seconds=2.0,
        )
        
        # Start watching and processing
        stats = live_processor.watch_and_process(
            input_dir=input_path,
            output_dir=output_path,
            pattern=pattern,
            ui=ui,
            num_workers=num_workers,
        )
        
        # Print summary
        print()  # Extra newline after counter
        ui.print_summary(processed=stats.total_processed, failed=stats.total_failed)
        
        if stats.total_failed > 0:
            ui.print_warning(f"{stats.total_failed} file(s) failed to process")
            for file_path, error in stats.failed_files[:5]:
                ui.print_error(f"{file_path.name}: {error}")
            if len(stats.failed_files) > 5:
                ui.print_error(f"... and {len(stats.failed_files) - 5} more failures")
        
        logger.info(f"Live mode complete: {stats.total_processed} processed, {stats.total_failed} failed")
        
        # Generate visualizations if requested
        if vis and stats.total_processed > 0:
            ui.print_info(f"Generating visualizations in: {vis}")
            limit_msg = f" (first {num_vis})" if num_vis else ""
            logger.info(f"Generating visualizations{limit_msg}")
            
            viz_success, viz_total = generate_visualizations(
                input_dir=input_dir,
                output_dir=output_dir,
                viz_dir=vis,
                prefix=prefix,
                pattern=pattern,
                show_progress=True,
                limit=num_vis,
                config=cfg,
            )
            logger.info(f"Visualizations: {viz_success}/{viz_total} created")
        
        # Exit with error code if any files failed
        if stats.total_failed > 0:
            sys.exit(1)
        
        return
    
    # NORMAL BATCH MODE
    # Count files first
    if recursive:
        files = list(input_path.rglob(pattern))
    else:
        files = list(input_path.glob(pattern))
    if recursive:
        files = list(input_path.rglob(pattern))
    else:
        files = list(input_path.glob(pattern))
    
    num_files = len(files)
    # For GPU: single worker is optimal (GPU handles parallelism)
    # For CPU: use multiple workers to parallelize across cores
    if jobs is not None:
        num_workers = jobs
    elif cpu:
        num_workers = max(1, (import_cpu_count() or 4) - 1)
    else:
        num_workers = 1  # GPU mode: single worker is optimal
    
    # Print configuration
    gpu_name = get_gpu_name() if not cpu else None
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
        limit_msg = f" (limit: {num_vis})" if num_vis else ""
        logger.info(f"Generating visualizations in: {vis}{limit_msg}")
        viz_success, viz_total = generate_visualizations(
            input_dir=input_dir,
            output_dir=output_dir,
            viz_dir=vis,
            prefix=prefix,
            pattern=pattern,
            show_progress=True,
            limit=num_vis,
            config=cfg,
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
