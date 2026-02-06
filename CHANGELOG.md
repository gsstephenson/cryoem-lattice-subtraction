# Changelog

All notable changes to lattice-sub will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.5] - 2026-02-06

### Added
- New `--suffix` option to filter files by suffix in batch/live mode
  - Example: `--suffix doseweighted.mrc` processes only `*doseweighted.mrc` files
  - Output uses standard naming (`sub_filename.mrc`)

## [1.5.4] - 2026-02-05

### Fixed
- Fixed unhandled ValueError when invalid pixel size values (negative or zero) are provided
  - Now shows user-friendly Click error instead of Python traceback
  - Uses `click.FloatRange(min=0.0, min_open=True)` for validation
- Fixed config file not overriding required `--pixel-size` option
  - `--pixel-size` is now optional when `--config` file provides `pixel_ang`
  - Command-line options still override config file values when both are provided

## [1.5.3] - 2026-02-03

### Fixed
- Fixed total file counter displaying incorrectly in live watch mode
- Fixed files added during batch processing not being queued for processing
- Improved edge case handling for empty directory starts in live mode

## [1.5.0] - 2026-02-03

### Added
- **Live Watch Mode**: New `--live` flag for continuous directory monitoring
  - Processes existing files first using multi-GPU batch mode
  - Automatically switches to single-GPU watch mode for new arrivals
  - File debouncing (2 seconds) to handle partial writes
  - Clean in-place counter: "Processed: X/Y files | Avg: Z.Zs/file | Latest: filename.mrc"
  - Graceful shutdown with Ctrl+C
- Added `watchdog>=3.0` dependency for file system monitoring
- New `LiveBatchProcessor` class for hybrid batch-then-watch processing
- New UI functions: `show_watch_startup()`, `show_watch_stopped()`, `update_live_counter()`

### Changed
- Live mode defaults to 1 worker (optimal for sequential file arrival)
- Batch mode retains multi-GPU automatic distribution for existing files
- GPU detection messages suppressed in live mode for clean counter display

## [1.4.0] - 2026-02-02

### Added
- Initial live mode implementation (superseded by v1.5.0 hybrid approach)

## [1.3.1] - 2026-01-15

### Changed
- Documentation improvements
- Minor bug fixes

## [1.3.0] - 2025-12-20

### Added
- Multi-GPU support with automatic workload distribution
- GPU status display at batch processing startup

### Changed
- Improved batch processing performance with parallel GPU execution

## [1.1.0] - 2025-11-15

### Added
- **Adaptive Per-Image Threshold Optimization**: Automatic grid search (1.40-1.60) to find optimal threshold per image
- **GPU-Accelerated Background Subtraction**: Kornia median_blur replaces scipy median_filter (48x speedup)
- Quality scoring function balancing peak count, SNR, distribution, and coverage

### Changed
- Default threshold changed from fixed (1.42) to **auto** (per-image optimization)
- Background subtraction moved to GPU (Kornia) from CPU (scipy)
- Overall processing speed improved ~5x with better per-image results

### Performance
- Time per image reduced from ~12s (v1.0.10) to ~2.6s (v1.1.0) with auto-optimization
- Single-image fixed threshold: ~1.0s (12x faster than v1.0.10)

## [1.0.10] - 2025-10-01

### Changed
- Stability improvements
- Fixed edge cases in FFT masking

## [1.0.0] - 2025-09-01

### Added
- Initial public release
- Command-line interface (`lattice-sub process`, `lattice-sub batch`)
- GPU acceleration via PyTorch
- Phase-preserving FFT inpainting algorithm
- YAML configuration file support
- Multi-panel visualization generation
- Python API (`LatticeSubtractor`, `BatchProcessor`)
- Pip-installable package

### Migration
- Modernized Python rewrite of legacy MATLAB code (LAsub.m)
- 10-100x performance improvement over MATLAB implementation
