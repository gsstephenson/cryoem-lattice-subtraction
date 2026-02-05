"""
Batch processing for multiple micrographs.

This module provides parallel processing capabilities for large datasets,
including automatic multi-GPU support for systems with multiple CUDA devices.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import logging

from tqdm import tqdm

from .config import Config
from .core import LatticeSubtractor
from .io import read_mrc, write_mrc


logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from batch processing."""
    
    total: int
    successful: int
    failed: int
    failed_files: List[Tuple[Path, str]]  # (path, error_message)
    
    @property
    def success_rate(self) -> float:
        """Return success rate as percentage."""
        return (self.successful / self.total) * 100 if self.total > 0 else 0.0


def _process_single_file(args: tuple) -> Tuple[Path, Optional[str]]:
    """
    Process a single file (for parallel execution).
    
    Args:
        args: Tuple of (input_path, output_path, config_dict)
        
    Returns:
        Tuple of (input_path, error_message or None)
    """
    input_path, output_path, config_dict = args
    
    try:
        # Reconstruct config from dict (can't pickle dataclass with defaults easily)
        config = Config(**config_dict)
        
        # Process
        subtractor = LatticeSubtractor(config)
        result = subtractor.process(input_path)
        result.save(output_path, pixel_size=config.pixel_ang)
        
        return (Path(input_path), None)
    
    except Exception as e:
        return (Path(input_path), str(e))


def _gpu_worker(
    gpu_id: int,
    file_pairs: List[Tuple[str, str]],
    config_dict: dict,
    progress_queue: mp.Queue,
    error_queue: mp.Queue,
):
    """
    Worker function for multi-GPU processing.
    
    Each worker processes its assigned files on a specific GPU and reports
    progress through a shared queue.
    
    Args:
        gpu_id: CUDA device ID to use
        file_pairs: List of (input_path, output_path) tuples
        config_dict: Configuration dictionary
        progress_queue: Queue to report progress (sends 1 for each completed file)
        error_queue: Queue to report errors (sends (gpu_id, file_path, error_msg))
    """
    import torch
    
    # Set this process to use the specific GPU
    torch.cuda.set_device(gpu_id)
    
    # Reconstruct config with the specific device_id and quiet mode
    config_dict = config_dict.copy()
    config_dict['device_id'] = gpu_id
    config_dict['_quiet'] = True  # Suppress messages - main process handles this
    config = Config(**config_dict)
    
    # Create subtractor (messages suppressed via _quiet flag)
    subtractor = LatticeSubtractor(config)
    
    for input_path, output_path in file_pairs:
        try:
            result = subtractor.process(input_path)
            result.save(output_path, pixel_size=config.pixel_ang)
            progress_queue.put(1)
        except Exception as e:
            error_queue.put((gpu_id, input_path, str(e)))
            return  # Fail-fast: exit on first error


def _check_gpu_memory(device_id: int, image_shape: Tuple[int, int]) -> Tuple[bool, str]:
    """
    Check if GPU has sufficient memory for processing.
    
    Args:
        device_id: CUDA device ID
        image_shape: (height, width) of image
        
    Returns:
        (is_ok, message) - True if sufficient memory, False with warning message
    """
    try:
        import torch
        free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        
        # Estimate memory needed: image + FFT (complex) + masks + overhead
        # Roughly 16x image size for safe margin (complex FFT, intermediate buffers)
        image_bytes = image_shape[0] * image_shape[1] * 4  # float32
        estimated_need = image_bytes * 16
        
        if free_mem < estimated_need:
            return False, (
                f"GPU {device_id}: {free_mem / 1e9:.1f}GB free, "
                f"need ~{estimated_need / 1e9:.1f}GB"
            )
        return True, ""
    except Exception as e:
        return True, ""  # If we can't check, proceed anyway


def _get_available_gpus() -> List[int]:
    """Get list of available CUDA GPU device IDs."""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    except ImportError:
        return []


class BatchProcessor:
    """
    Parallel batch processor for micrograph datasets.
    
    This class handles processing of multiple MRC files in parallel,
    with progress tracking, error handling, and optional file pattern matching.
    
    Example:
        >>> config = Config(pixel_ang=0.56)
        >>> processor = BatchProcessor(config, num_workers=8)
        >>> result = processor.process_directory(
        ...     input_dir="raw_micrographs/",
        ...     output_dir="subtracted/",
        ...     pattern="*.mrc"
        ... )
        >>> print(f"Processed {result.successful}/{result.total} files")
    """
    
    def __init__(
        self,
        config: Config,
        num_workers: Optional[int] = None,
        output_prefix: str = "sub_",
    ):
        """
        Initialize batch processor.
        
        Args:
            config: Processing configuration
            num_workers: Number of parallel workers. Default: CPU count - 1
            output_prefix: Prefix for output filenames. Default: "sub_"
        """
        self.config = config
        self.num_workers = num_workers or max(1, os.cpu_count() - 1)
        self.output_prefix = output_prefix
        
        # Convert config to dict for pickling
        from dataclasses import asdict
        self._config_dict = asdict(config)
    
    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.mrc",
        recursive: bool = False,
        show_progress: bool = True,
    ) -> BatchResult:
        """
        Process all matching files in a directory.
        
        Args:
            input_dir: Input directory containing MRC files
            output_dir: Output directory for processed files
            pattern: Glob pattern for matching files. Default: "*.mrc"
            recursive: If True, search subdirectories. Default: False
            show_progress: If True, show progress bar. Default: True
            
        Returns:
            BatchResult with processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Find input files
        if recursive:
            input_files = list(input_dir.rglob(pattern))
        else:
            input_files = list(input_dir.glob(pattern))
        
        if not input_files:
            logger.warning(f"No files matching '{pattern}' found in {input_dir}")
            return BatchResult(total=0, successful=0, failed=0, failed_files=[])
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build file list with output paths
        file_pairs = []
        for input_path in input_files:
            output_name = f"{self.output_prefix}{input_path.name}"
            output_path = output_dir / output_name
            file_pairs.append((input_path, output_path))
        
        return self.process_file_list(file_pairs, show_progress=show_progress)
    
    def process_file_list(
        self,
        file_pairs: List[Tuple[Path, Path]],
        show_progress: bool = True,
    ) -> BatchResult:
        """
        Process a list of input/output file pairs.
        
        Automatically uses multi-GPU processing when multiple GPUs are available.
        Files are distributed evenly across GPUs in chunks.
        
        Args:
            file_pairs: List of (input_path, output_path) tuples
            show_progress: If True, show progress bar
            
        Returns:
            BatchResult with processing statistics
        """
        total = len(file_pairs)
        successful = 0
        failed_files = []
        
        # Check if using GPU - if so, check for multi-GPU capability
        use_gpu = self.config.backend == "pytorch"
        if self.config.backend == "auto":
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
        
        if use_gpu:
            # Check how many GPUs are available
            available_gpus = _get_available_gpus()
            
            if len(available_gpus) > 1 and total > 1:
                # Print GPU list
                try:
                    import torch
                    from .ui import get_ui, Colors
                    ui = get_ui(quiet=self.config._quiet)
                    print()
                    for gpu_id in available_gpus:
                        gpu_name = torch.cuda.get_device_name(gpu_id)
                        print(f"  {ui._colorize('✓', Colors.GREEN)} GPU {gpu_id}: {gpu_name}")
                    print()
                except Exception:
                    pass
                
                # Multi-GPU processing
                successful, failed_files = self._process_multi_gpu(
                    file_pairs, available_gpus, show_progress
                )
            else:
                # Single GPU - sequential processing
                successful, failed_files = self._process_sequential(
                    file_pairs, show_progress
                )
        else:
            # Parallel processing for CPU
            successful, failed_files = self._process_parallel(
                file_pairs, show_progress
            )
        
        return BatchResult(
            total=total,
            successful=successful,
            failed=total - successful,
            failed_files=failed_files,
        )
    
    def _process_sequential(
        self,
        file_pairs: List[Tuple[Path, Path]],
        show_progress: bool = True,
    ) -> Tuple[int, List[Tuple[Path, str]]]:
        """Process files sequentially (for GPU mode)."""
        import sys
        successful = 0
        failed_files = []
        
        # Create progress bar FIRST before any CUDA initialization
        # Use sys.stdout and force flush for immediate display
        if show_progress:
            print("", flush=True)  # Ensure clean line
            pbar = tqdm(
                total=len(file_pairs),
                desc="  Processing",
                unit="file",
                ncols=80,
                leave=True,
            )
        else:
            pbar = None
        
        # Now initialize subtractor (this triggers CUDA init)
        subtractor = LatticeSubtractor(self.config)
        
        for input_path, output_path in file_pairs:
            try:
                result = subtractor.process(input_path)
                result.save(output_path, pixel_size=self.config.pixel_ang)
                successful += 1
            except Exception as e:
                failed_files.append((input_path, str(e)))
                logger.error(f"Failed to process {input_path}: {e}")
            
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        return successful, failed_files
    
    def _process_parallel(
        self,
        file_pairs: List[Tuple[Path, Path]],
        show_progress: bool = True,
    ) -> Tuple[int, List[Tuple[Path, str]]]:
        """Process files in parallel (for CPU mode)."""
        successful = 0
        failed_files = []
        total = len(file_pairs)
        
        # Prepare arguments for parallel execution
        args_list = [
            (str(inp), str(out), self._config_dict)
            for inp, out in file_pairs
        ]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(_process_single_file, args): args[0]
                for args in args_list
            }
            
            # Track progress
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=total,
                    desc="Processing micrographs",
                    unit="file",
                )
            
            for future in iterator:
                input_path, error = future.result()
                
                if error is None:
                    successful += 1
                else:
                    failed_files.append((input_path, error))
                    logger.error(f"Failed to process {input_path}: {error}")
        
        return successful, failed_files
    
    def _process_multi_gpu(
        self,
        file_pairs: List[Tuple[Path, Path]],
        gpu_ids: List[int],
        show_progress: bool = True,
    ) -> Tuple[int, List[Tuple[Path, str]]]:
        """
        Process files in parallel across multiple GPUs.
        
        Files are distributed evenly across GPUs in chunks.
        Uses spawn-based multiprocessing to avoid CUDA fork issues.
        
        Args:
            file_pairs: List of (input_path, output_path) tuples
            gpu_ids: List of CUDA device IDs to use
            show_progress: If True, show unified progress bar
            
        Returns:
            (successful_count, failed_files_list)
        """
        import time
        
        total = len(file_pairs)
        num_gpus = len(gpu_ids)
        
        # Check GPU memory on first GPU (assume similar for all)
        if file_pairs:
            try:
                sample_image = read_mrc(file_pairs[0][0])
                is_ok, msg = _check_gpu_memory(gpu_ids[0], sample_image.shape)
                if not is_ok:
                    print(f"⚠ Memory warning: {msg}")
            except Exception:
                pass  # Proceed anyway
        
        # Distribute files evenly across GPUs (chunked distribution)
        chunk_size = (total + num_gpus - 1) // num_gpus  # Ceiling division
        gpu_file_assignments = []
        
        for i, gpu_id in enumerate(gpu_ids):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total)
            if start_idx < total:
                chunk = [(str(inp), str(out)) for inp, out in file_pairs[start_idx:end_idx]]
                gpu_file_assignments.append((gpu_id, chunk))
        
        # Create shared queues for progress and errors
        # Use 'spawn' context to avoid CUDA fork issues
        ctx = mp.get_context('spawn')
        progress_queue = ctx.Queue()
        error_queue = ctx.Queue()
        
        # Create progress bar (after all GPU info printed)
        if show_progress:
            print()  # Blank line for visual separation
            pbar = tqdm(
                total=total,
                desc="  Processing",
                unit="file",
                ncols=80,
                leave=True,
            )
        else:
            pbar = None
        
        # Start worker processes
        processes = []
        for gpu_id, file_chunk in gpu_file_assignments:
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, file_chunk, self._config_dict, progress_queue, error_queue),
            )
            p.start()
            processes.append(p)
        
        # Monitor progress and check for errors
        successful = 0
        failed_files = []
        completed = 0
        
        while completed < total:
            # Check for progress updates (non-blocking with timeout)
            try:
                while True:
                    progress_queue.get(timeout=0.1)
                    successful += 1
                    completed += 1
                    if pbar:
                        pbar.update(1)
            except:
                pass  # Queue empty, continue
            
            # Check for errors (non-blocking)
            try:
                while True:
                    gpu_id, file_path, error_msg = error_queue.get_nowait()
                    failed_files.append((Path(file_path), error_msg))
                    completed += 1
                    if pbar:
                        pbar.update(1)
                    
                    # Fail-fast: terminate all workers and report
                    print(f"\n✗ GPU {gpu_id} failed on {Path(file_path).name}: {error_msg}")
                    print(f"\nTip: Try a different configuration:")
                    print(f"  lattice-sub batch <input> <output> -p {self.config.pixel_ang} --cpu -j 8")
                    
                    # Terminate all processes
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    
                    if pbar:
                        pbar.close()
                    
                    return successful, failed_files
            except:
                pass  # No errors, continue
            
            # Check if all processes have finished
            all_done = all(not p.is_alive() for p in processes)
            if all_done:
                # Drain remaining queue items
                try:
                    while True:
                        progress_queue.get_nowait()
                        successful += 1
                        completed += 1
                        if pbar:
                            pbar.update(1)
                except:
                    pass
                break
        
        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=1.0)
        
        if pbar:
            pbar.close()
        
        return successful, failed_files

    def process_numbered_sequence(
        self,
        input_pattern: str,
        output_dir: str | Path,
        start: int,
        end: int,
        zero_pad: int = 4,
        show_progress: bool = True,
    ) -> BatchResult:
        """
        Process a numbered sequence of files.
        
        This is designed to match the legacy HYPER_loop behavior for 
        processing numbered file sequences.
        
        Args:
            input_pattern: Pattern with {num} placeholder, e.g., 
                          "raw/image_{num}.mrc"
            output_dir: Output directory
            start: Starting number (inclusive)
            end: Ending number (inclusive)
            zero_pad: Number of digits for zero-padding. Default: 4
            show_progress: If True, show progress bar
            
        Returns:
            BatchResult
            
        Example:
            >>> processor.process_numbered_sequence(
            ...     input_pattern="data/mic_{num}.mrc",
            ...     output_dir="processed/",
            ...     start=1,
            ...     end=100,
            ...     zero_pad=4
            ... )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_pairs = []
        
        for num in range(start, end + 1):
            num_str = str(num).zfill(zero_pad)
            input_path = Path(input_pattern.format(num=num_str))
            
            if input_path.exists():
                output_name = f"{self.output_prefix}{input_path.name}"
                output_path = output_dir / output_name
                file_pairs.append((input_path, output_path))
            else:
                logger.debug(f"Skipping non-existent file: {input_path}")
        
        if not file_pairs:
            logger.warning("No files found matching the numbered pattern")
            return BatchResult(total=0, successful=0, failed=0, failed_files=[])
        
        return self.process_file_list(file_pairs, show_progress=show_progress)


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    config: Config,
    pattern: str = "*.mrc",
    num_workers: Optional[int] = None,
    show_progress: bool = True,
) -> BatchResult:
    """
    Convenience function for batch processing a directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        config: Processing configuration
        pattern: Glob pattern for files
        num_workers: Number of parallel workers
        show_progress: Show progress bar
        
    Returns:
        BatchResult
    """
    processor = BatchProcessor(config, num_workers=num_workers)
    return processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
        show_progress=show_progress,
    )
