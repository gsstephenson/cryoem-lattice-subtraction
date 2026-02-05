"""
Live watch mode for processing files as they arrive.

This module provides functionality for monitoring a directory and
processing MRC files as they are created/modified, enabling real-time
processing pipelines (e.g., from motion correction output).
"""

import logging
import time
import threading
from pathlib import Path
from typing import Set, Dict, Optional, List, Tuple
from queue import Queue, Empty
from dataclasses import dataclass

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from .config import Config
from .core import LatticeSubtractor
from .io import write_mrc
from .ui import TerminalUI

logger = logging.getLogger(__name__)


@dataclass
class LiveStats:
    """Statistics for live processing."""
    
    total_processed: int = 0
    total_failed: int = 0
    total_time: float = 0.0
    failed_files: List[Tuple[Path, str]] = None
    
    def __post_init__(self):
        if self.failed_files is None:
            self.failed_files = []
    
    @property
    def average_time(self) -> float:
        """Get average processing time per file."""
        if self.total_processed == 0:
            return 0.0
        return self.total_time / self.total_processed
    
    def add_success(self, processing_time: float):
        """Record a successful processing."""
        self.total_processed += 1
        self.total_time += processing_time
    
    def add_failure(self, file_path: Path, error: str):
        """Record a failed processing."""
        self.total_failed += 1
        self.failed_files.append((file_path, error))


class MRCFileHandler(FileSystemEventHandler):
    """
    Handles file system events for MRC files.
    
    Implements debouncing to ensure files are completely written before
    processing. Files are added to a processing queue after being stable
    for a specified duration.
    """
    
    def __init__(
        self,
        pattern: str,
        file_queue: Queue,
        processed_files: Set[Path],
        processor: 'LiveBatchProcessor',
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize file handler.
        
        Args:
            pattern: Glob pattern for matching files (e.g., "*.mrc")
            file_queue: Queue to add detected files to
            processed_files: Set of already processed file paths
            processor: Parent LiveBatchProcessor for updating totals
            debounce_seconds: Time to wait after last modification before processing
        """
        super().__init__()
        self.pattern = pattern
        self.file_queue = file_queue
        self.processed_files = processed_files
        self.processor = processor
        self.debounce_seconds = debounce_seconds
        
        # Track file modification times for debouncing
        self._pending_files: Dict[Path, float] = {}
        self._lock = threading.Lock()
        
        # Start debounce checker thread
        self._running = True
        self._checker_thread = threading.Thread(target=self._check_pending_files, daemon=True)
        self._checker_thread.start()
    
    def stop(self):
        """Stop the debounce checker thread."""
        self._running = False
        if self._checker_thread.is_alive():
            self._checker_thread.join(timeout=5.0)
    
    def _matches_pattern(self, path: Path) -> bool:
        """Check if file matches the pattern."""
        import fnmatch
        return fnmatch.fnmatch(path.name, self.pattern)
    
    def _check_pending_files(self):
        """Background thread to check for stable files ready to process."""
        while self._running:
            time.sleep(0.5)  # Check every 0.5 seconds
            
            current_time = time.time()
            files_to_queue = []
            
            with self._lock:
                # Find files that are stable (no modifications for debounce_seconds)
                for file_path, last_mod in list(self._pending_files.items()):
                    if current_time - last_mod >= self.debounce_seconds:
                        files_to_queue.append(file_path)
                        del self._pending_files[file_path]
            
            # Queue stable files
            for file_path in files_to_queue:
                if file_path not in self.processed_files and file_path.exists():
                    logger.debug(f"Queueing stable file: {file_path}")
                    self.file_queue.put(file_path)
                    # Increment total count when new file is queued
                    with self.processor._lock:
                        self.processor.total_files += 1
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if self._matches_pattern(file_path) and file_path not in self.processed_files:
            logger.debug(f"File created: {file_path}")
            with self._lock:
                self._pending_files[file_path] = time.time()
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if self._matches_pattern(file_path) and file_path not in self.processed_files:
            logger.debug(f"File modified: {file_path}")
            with self._lock:
                self._pending_files[file_path] = time.time()


class LiveBatchProcessor:
    """
    Live batch processor that watches a directory and processes files as they arrive.
    
    This processor monitors an input directory for new MRC files and processes them
    in real-time as they are created (e.g., from motion correction output).
    
    Features:
    - File debouncing to ensure complete writes
    - Real-time progress counter (instead of progress bar)
    - Resilient error handling (continues on failures)
    - Support for multi-GPU, single-GPU, and CPU processing
    - Deferred visualization generation (after watching stops)
    """
    
    def __init__(
        self,
        config: Config,
        output_prefix: str = "sub_",
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize live batch processor.
        
        Args:
            config: Processing configuration
            output_prefix: Prefix for output filenames
            debounce_seconds: Time to wait after file modification before processing
        """
        self.config = config
        self.output_prefix = output_prefix
        self.debounce_seconds = debounce_seconds
        
        # Processing state
        self.file_queue: Queue = Queue()
        self.processed_files: Set[Path] = set()
        self.stats = LiveStats()
        self.total_files: int = 0  # Total files in input directory
        
        # Worker threads
        self._workers: List[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()
        
        # File system watcher
        self.observer: Optional[Observer] = None
        self.handler: Optional[MRCFileHandler] = None
    
    def _create_subtractor(self, device_id: Optional[int] = None) -> LatticeSubtractor:
        """Create a LatticeSubtractor instance with optional device override."""
        if device_id is not None:
            # Create config copy with specific device
            from dataclasses import replace
            config = replace(self.config, device_id=device_id)
        else:
            config = self.config
        
        # Enable quiet mode to suppress GPU messages on each file
        config._quiet = True
        
        return LatticeSubtractor(config)
    
    def _process_worker(
        self,
        output_dir: Path,
        ui: TerminalUI,
        device_id: Optional[int] = None,
    ):
        """
        Worker thread that processes files from the queue.
        
        Args:
            output_dir: Output directory for processed files
            ui: Terminal UI for displaying progress
            device_id: Optional GPU device ID (for multi-GPU)
        """
        # Set CUDA device for this thread if GPU is being used
        if device_id is not None:
            import torch
            torch.cuda.set_device(device_id)
            logger.debug(f"Worker initialized on GPU {device_id}")
        
        while self._running:
            try:
                # Get file from queue with timeout
                file_path = self.file_queue.get(timeout=0.5)
            except Empty:
                continue
            
            # Create subtractor on-demand for each file to avoid memory buildup
            subtractor = self._create_subtractor(device_id)
            
            # Process the file
            output_name = f"{self.output_prefix}{file_path.name}"
            output_path = output_dir / output_name
            
            start_time = time.time()
            
            try:
                result = subtractor.process(file_path)
                result.save(output_path, pixel_size=self.config.pixel_ang)
                
                processing_time = time.time() - start_time
                
                with self._lock:
                    self.stats.add_success(processing_time)
                    self.processed_files.add(file_path)
                
                # Update UI counter
                ui.update_live_counter(
                    count=self.stats.total_processed,
                    total=self.total_files,
                    avg_time=self.stats.average_time,
                    latest=file_path.name,
                )
                
                # Don't log to console in live mode - it breaks the in-place counter update
            
            except Exception as e:
                with self._lock:
                    self.stats.add_failure(file_path, str(e))
                    self.processed_files.add(file_path)  # Don't retry
                
                logger.error(f"Failed to process {file_path.name}: {e}")
            
            finally:
                # Clean up memory after each file in live mode
                del subtractor
                if device_id is not None:
                    import torch
                    torch.cuda.empty_cache()
                
                self.file_queue.task_done()
    
    def watch_and_process(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str,
        ui: TerminalUI,
        num_workers: int = 1,
    ) -> LiveStats:
        """
        Start watching directory and processing files as they arrive.
        
        This method blocks until KeyboardInterrupt (Ctrl+C) is received.
        
        Args:
            input_dir: Directory to watch for new files
            output_dir: Output directory for processed files
            pattern: Glob pattern for matching files (e.g., "*.mrc")
            ui: Terminal UI for displaying progress
            num_workers: Number of processing workers (for multi-GPU or CPU)
            
        Returns:
            LiveStats with processing statistics
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing files in directory
        all_files = list(input_dir.glob(pattern))
        
        # If files already exist, process them with batch mode first (multi-GPU)
        if all_files:
            from .batch import BatchProcessor
            
            ui.print_info(f"Found {len(all_files)} existing files - processing with batch mode first")
            
            # Create file pairs for batch processing
            file_pairs = []
            for file_path in all_files:
                output_name = f"{self.output_prefix}{file_path.name}"
                output_path = output_dir / output_name
                file_pairs.append((file_path, output_path))
                self.processed_files.add(file_path)  # Mark as processed
            
            # Process with BatchProcessor (will use multi-GPU if available)
            batch_processor = BatchProcessor(
                config=self.config,
                num_workers=num_workers,
                output_prefix="",  # Already included in output paths
            )
            
            result = batch_processor.process_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                pattern=pattern,
                recursive=False,
                show_progress=True,
            )
            
            # Update stats with batch results
            self.stats.total_processed = result.successful
            self.stats.total_failed = result.failed
            self.total_files = len(all_files)  # Set initial total
            
            ui.print_info(f"Batch processing complete: {result.successful}/{result.total} files")
            print()
            
            # Check if any new files arrived during batch processing
            current_files = set(input_dir.glob(pattern))
            new_during_batch = current_files - set(all_files)
            if new_during_batch:
                ui.print_info(f"Found {len(new_during_batch)} files added during batch processing - queueing now")
                for file_path in new_during_batch:
                    if file_path not in self.processed_files:
                        self.file_queue.put(file_path)
                        self.total_files += 1  # Increment total for each new file
        else:
            # No existing files, start fresh
            self.total_files = 0
        
        # Setup file system watcher
        self.handler = MRCFileHandler(
            pattern=pattern,
            file_queue=self.file_queue,
            processed_files=self.processed_files,
            processor=self,
            debounce_seconds=self.debounce_seconds,
        )
        
        self.observer = Observer()
        self.observer.schedule(self.handler, str(input_dir), recursive=False)
        self.observer.start()
        
        # Determine GPU setup for workers
        device_ids = self._get_worker_devices(num_workers)
        
        # Print GPU list at startup (non-dynamic, just info)
        if device_ids and device_ids[0] is not None:
            try:
                import torch
                from .ui import Colors
                unique_gpus = sorted(set(d for d in device_ids if d is not None))
                print()
                for gpu_id in unique_gpus:
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    print(f"  {ui._colorize('✓', Colors.GREEN)} GPU {gpu_id}: {gpu_name}")
                print()
            except Exception as e:
                pass  # Silently skip if GPU info unavailable
        
        # Start processing workers
        self._running = True
        for i, device_id in enumerate(device_ids):
            worker = threading.Thread(
                target=self._process_worker,
                args=(output_dir, ui, device_id),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        
        # Show initial counter
        ui.show_live_counter_header()
        ui.update_live_counter(count=0, total=self.total_files, avg_time=0.0, latest="waiting...")
        
        # Wait for interrupt
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            ui.show_watch_stopped()
        
        # Cleanup
        self._shutdown(ui)
        
        return self.stats
    
    def _get_worker_devices(self, num_workers: int) -> List[Optional[int]]:
        """
        Determine device IDs for workers based on available GPUs.
        
        Returns:
            List of device IDs (None for CPU workers)
        """
        # Check if CPU-only mode is forced
        if self.config.backend == "numpy":
            return [None] * num_workers
        
        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                
                if gpu_count > 1 and num_workers > 1:
                    # Multi-GPU: assign workers to GPUs in round-robin
                    return [i % gpu_count for i in range(num_workers)]
                else:
                    # Single GPU: all workers use GPU 0
                    return [0] * num_workers
        except ImportError:
            pass
        
        # CPU mode: all workers use None (CPU)
        return [None] * num_workers
    
    def _shutdown(self, ui: TerminalUI):
        """Shutdown workers and observer cleanly."""
        # Stop accepting new files
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5.0)
        
        if self.handler:
            self.handler.stop()
        
        # Wait for queue to be processed
        ui.print_info("Processing remaining queued files...")
        self.file_queue.join()
        
        # Stop workers
        self._running = False
        for worker in self._workers:
            worker.join(timeout=5.0)
