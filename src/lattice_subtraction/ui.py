"""
Terminal UI utilities for lattice subtraction.

This module provides styled terminal output with ASCII art banner
and formatted progress messages. Output is only shown when running
interactively (TTY detected) and not suppressed by --quiet flag.

When piped or used in a pipeline, decorative output is automatically
suppressed to avoid polluting downstream processing.
"""

import sys
import time
from typing import Optional, List


class Colors:
    """ANSI color codes for terminal styling."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


# ASCII Art Banner
BANNER = r"""
.__          __    __  .__                                   ___.    
|  | _____ _/  |__/  |_|__| ____  ____             ________ _\_ |__  
|  | \__  \\   __\   __\  |/ ___\/ __ \   ______  /  ___/  |  \ __ \ 
|  |__/ __ \|  |  |  | |  \  \__\  ___/  /_____/  \___ \|  |  / \_\ \
|____(____  /__|  |__| |__|\___  >___  >         /____  >____/|___  /
          \/                   \/    \/               \/          \/ 
"""

# Import version from package to keep it in sync
from . import __version__ as VERSION


def is_interactive() -> bool:
    """
    Check if running in an interactive terminal.
    
    Returns False if stdout is piped or redirected, which means
    we're likely part of a pipeline and should suppress decorative output.
    """
    return sys.stdout.isatty()


class TerminalUI:
    """
    Manages styled terminal output for the CLI.
    
    Decorative output (banner, colors, progress indicators) is only shown when:
    - Running in an interactive terminal (TTY detected)
    - Not suppressed by quiet mode
    
    When piped or in a script, output is automatically minimal.
    """
    
    def __init__(self, quiet: bool = False):
        """
        Initialize the terminal UI.
        
        Args:
            quiet: If True, suppress all decorative output even in interactive mode
        """
        self.quiet = quiet
        self.interactive = is_interactive() and not quiet
        self.use_colors = self.interactive
        self._start_time: Optional[float] = None
        self._file_start_time: Optional[float] = None
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text
    
    def print_banner(self) -> None:
        """Print the ASCII art banner."""
        if not self.interactive:
            return
        
        print()
        print(self._colorize(BANNER, Colors.CYAN))
        tagline = f"  Phase-preserving FFT inpainting for cryo-EM  |  v{VERSION}"
        print(self._colorize(tagline, Colors.DIM))
        print()
    
    def print_config(self, pixel_size: float, threshold: float, 
                     backend: str, gpu_name: Optional[str] = None) -> None:
        """Print configuration summary."""
        if not self.interactive:
            return
        
        print(self._colorize("  Configuration", Colors.BOLD))
        print(self._colorize("  -------------", Colors.DIM))
        print(f"    Pixel size:  {self._colorize(f'{pixel_size} A', Colors.YELLOW)}")
        print(f"    Threshold:   {self._colorize(str(threshold), Colors.YELLOW)}")
        
        # Determine backend display string
        if backend == "auto":
            # Check if GPU is actually available for auto mode
            if gpu_name:
                backend_str = f"Auto → GPU ({gpu_name})"
                print(f"    Backend:     {self._colorize(backend_str, Colors.GREEN)}")
            else:
                backend_str = "Auto → CPU"
                print(f"    Backend:     {self._colorize(backend_str, Colors.BLUE)}")
        elif backend == "pytorch" and gpu_name:
            backend_str = f"PyTorch CUDA ({gpu_name})"
            print(f"    Backend:     {self._colorize(backend_str, Colors.GREEN)}")
        elif backend == "pytorch":
            print(f"    Backend:     {self._colorize('PyTorch CUDA', Colors.GREEN)}")
        else:
            print(f"    Backend:     {self._colorize('NumPy (CPU)', Colors.BLUE)}")
        print()
    
    def start_timer(self) -> None:
        """Start the overall timer."""
        self._start_time = time.time()
    
    def start_processing(self, filename: str, shape: Optional[tuple] = None) -> None:
        """Indicate start of processing a file."""
        self._file_start_time = time.time()
        
        if not self.interactive:
            return
        
        print(f"  {self._colorize('>', Colors.CYAN)} Processing: {self._colorize(filename, Colors.BOLD)}")
        if shape:
            print(f"    {self._colorize('|-', Colors.DIM)} Size: {shape[0]} x {shape[1]}")
    
    def end_processing(self, output_path: str, success: bool = True) -> None:
        """Indicate end of processing."""
        elapsed = time.time() - self._file_start_time if self._file_start_time else 0
        
        if not self.interactive:
            return
        
        if success:
            status = self._colorize(f"[OK] Complete ({elapsed:.2f}s)", Colors.GREEN)
        else:
            status = self._colorize("[FAIL]", Colors.RED)
        
        print(f"    {self._colorize('`-', Colors.DIM)} {status}")
        print()
    
    def print_batch_header(self, num_files: int, output_dir: str, num_workers: int = 1) -> None:
        """Print batch processing header."""
        if not self.interactive:
            return
        
        print(self._colorize("  Batch Processing", Colors.BOLD))
        print(self._colorize("  ----------------", Colors.DIM))
        print(f"    Files:    {self._colorize(str(num_files), Colors.YELLOW)}")
        print(f"    Output:   {output_dir}")
        print(f"    Workers:  {num_workers}")
        print()
    
    def print_batch_progress(self, current: int, total: int, filename: str, 
                             elapsed: Optional[float] = None) -> None:
        """Print batch progress update."""
        if not self.interactive:
            return
        
        progress = f"[{current}/{total}]"
        time_str = f" ({elapsed:.1f}s)" if elapsed else ""
        print(f"  {self._colorize(progress, Colors.CYAN)} {filename}{time_str}")
    
    def print_batch_complete(self) -> None:
        """Print batch completion message."""
        if not self.interactive:
            return
        
        elapsed = time.time() - self._start_time if self._start_time else 0
        print()
        print(f"  {self._colorize('[OK]', Colors.GREEN)} {self._colorize('Batch complete', Colors.BOLD)} ({elapsed:.1f}s)")
        print()
    
    def print_summary(self, processed: int, failed: int = 0) -> None:
        """Print final summary."""
        if not self.interactive:
            return
        
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        print(self._colorize("  Summary", Colors.BOLD))
        print(self._colorize("  -------", Colors.DIM))
        print(f"    Processed: {self._colorize(str(processed), Colors.GREEN)}")
        if failed > 0:
            print(f"    Failed:    {self._colorize(str(failed), Colors.RED)}")
        print(f"    Time:      {elapsed:.1f}s")
        print()
    
    def print_error(self, message: str) -> None:
        """Print error message (always shown unless quiet)."""
        if self.quiet:
            return
        prefix = self._colorize("Error:", Colors.RED) if self.use_colors else "Error:"
        print(f"  {prefix} {message}", file=sys.stderr)
    
    def print_warning(self, message: str) -> None:
        """Print warning message."""
        if not self.interactive:
            return
        prefix = self._colorize("Warning:", Colors.YELLOW)
        print(f"  {prefix} {message}", file=sys.stderr)
    
    def print_success(self, message: str) -> None:
        """Print success message."""
        if not self.interactive:
            return
        check = self._colorize("[OK]", Colors.GREEN)
        print(f"  {check} {message}")
    
    def print_info(self, message: str) -> None:
        """Print info message."""
        if not self.interactive:
            return
        print(f"  {self._colorize('>', Colors.CYAN)} {message}")
    
    def print_saved(self, path: str) -> None:
        """Print saved file notification."""
        if not self.interactive:
            return
        print(f"    {self._colorize('|-', Colors.DIM)} Saved: {path}")
    
    def show_watch_startup(self, input_dir: str) -> None:
        """Show live watch mode startup message."""
        if not self.interactive:
            return
        
        print(self._colorize("  Live Watch Mode", Colors.BOLD))
        print(self._colorize("  ---------------", Colors.DIM))
        print(f"    Watching: {input_dir}")
        print(f"    {self._colorize('Press Ctrl+C to stop watching and finalize', Colors.YELLOW)}")
        print()
    
    def show_watch_stopped(self) -> None:
        """Show message when watching is stopped."""
        if not self.interactive:
            return
        print()  # New line after counter
        print(f"  {self._colorize('[STOPPED]', Colors.YELLOW)} Watch mode stopped")
        print()
    
    def show_live_counter_header(self) -> None:
        """Show header for live counter display."""
        if not self.interactive:
            return
        # No header needed - counter updates in place
        pass
    
    def update_live_counter(self, count: int, avg_time: float, latest: str) -> None:
        """
        Update the live processing counter in place.
        
        Args:
            count: Number of files processed
            avg_time: Average processing time per file
            latest: Name of most recently processed file
        """
        if not self.interactive:
            return
        
        # Format time display
        if avg_time > 0:
            time_str = f"{avg_time:.1f}s/file"
        else:
            time_str = "--s/file"
        
        # Truncate filename if too long
        max_filename_len = 40
        if len(latest) > max_filename_len:
            latest = "..." + latest[-(max_filename_len-3):]
        
        # Build counter line
        counter = f"  Processed: {self._colorize(str(count), Colors.GREEN)} files"
        avg = f"Avg: {self._colorize(time_str, Colors.CYAN)}"
        file_info = f"Latest: {latest}"
        
        line = f"{counter} | {avg} | {file_info}"
        
        # Print with carriage return to overwrite previous line
        print(f"\r{line}", end="", flush=True)
    
    def print_gpu_status(self, gpu_ids: List[int], gpu_names: List[str]) -> None:
        """
        Print initial GPU status display.
        
        Args:
            gpu_ids: List of GPU device IDs
            gpu_names: List of GPU names
        """
        if not self.interactive:
            return
        
        print()
        for gpu_id, name in zip(gpu_ids, gpu_names):
            print(f"  {self._colorize('✓', Colors.GREEN)} GPU {gpu_id}: {name}")
        print()


def get_ui(quiet: bool = False) -> TerminalUI:
    """
    Get a TerminalUI instance.
    
    Args:
        quiet: If True, suppress decorative output even in interactive mode
        
    Returns:
        TerminalUI instance configured for current environment
    """
    return TerminalUI(quiet=quiet)


def get_gpu_name() -> Optional[str]:
    """Get the name of the CUDA GPU if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None
