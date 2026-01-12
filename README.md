# Lattice Subtraction for Cryo-EM Micrographs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for computationally removing periodic crystal lattice signals (Bragg spots) from cryo-EM micrographs to reveal non-periodic features such as defects, individual particles, or molecular tags.

## Overview

When imaging 2D crystal samples in cryo-EM, the periodic lattice produces strong diffraction spots in Fourier space that can obscure weaker signals from non-periodic features. This package implements a **phase-preserving lattice subtraction algorithm** that:

1. Identifies lattice peaks via thresholding on the log-power spectrum
2. Protects low-frequency (structural) and high-frequency (noise) regions
3. **Inpaints** the lattice peak regions with local average amplitudes
4. Preserves the original phase information for accurate reconstruction

This is a modernized Python port of legacy MATLAB code, with added features:
- **10-100× faster** processing via optimized NumPy/SciPy operations
- **Optional GPU acceleration** using CuPy
- **Parallel batch processing** for large datasets
- **Configuration files** (YAML) for reproducible experiments
- **Command-line interface** for easy integration into pipelines

## Installation

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/kasinath-lab/lattice-subtraction.git
cd lattice-subtraction

# Create and activate conda environment
conda env create -f environment.yml
conda activate lattice_sub

# Install in development mode
pip install -e .
```

### Quick Install

```bash
pip install lattice-subtraction
```

### GPU Support (Optional)

For CUDA GPU acceleration:

```bash
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install cupy-cuda11x  # For CUDA 11.x
```

## Quick Start

### Python API

```python
from lattice_subtraction import LatticeSubtractor, Config

# Create configuration
config = Config(
    pixel_ang=0.56,      # Pixel size in Angstroms
    threshold=1.56,      # Peak detection threshold
    inside_radius_ang=90 # Protect low-frequency region
)

# Process a micrograph
subtractor = LatticeSubtractor(config)
result = subtractor.process("input.mrc")
result.save("output.mrc")
```

### Command Line

```bash
# Process a single file
lattice-sub process input.mrc -o output.mrc --pixel-size 0.56

# Batch process a directory (8 parallel workers)
lattice-sub batch input_dir/ output_dir/ --pixel-size 0.56 -j 8

# Create a configuration file
lattice-sub init-config params.yaml --pixel-size 0.56 --detector K3
```

### Using Configuration Files

```yaml
# params.yaml
pixel_ang: 0.56
threshold: 1.56
inside_radius_ang: 90
outside_radius_ang: 1.32  # or "auto"
expand_pixel: 10
unit_cell_ang: 116  # Nucleosome repeat distance
backend: numpy  # or "cupy" for GPU
```

```bash
lattice-sub process input.mrc -o output.mrc --config params.yaml
```

## Algorithm Details

### Processing Pipeline

```
Input Image → Pad → FFT → Detect Peaks → Create Mask → Inpaint → iFFT → Crop → Output
```

1. **Padding**: Image is padded with mean value to reduce edge artifacts
2. **FFT**: 2D Fast Fourier Transform brings image to frequency domain
3. **Peak Detection**: Log-power spectrum is background-subtracted and thresholded
4. **Mask Creation**: Combines peak mask with radial limits:
   - **Inner radius** (default 90Å): Protects low-frequency structural information
   - **Outer radius** (near Nyquist): Protects high-frequency details
5. **Inpainting**: Lattice peaks are replaced with local average amplitude from 4 neighboring pixels (shifted by ~half unit cell distance)
6. **Phase Preservation**: Original phase is retained; only amplitude is modified
7. **Inverse FFT**: Returns to real space with lattice removed

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pixel_ang` | *required* | Pixel size in Ångstroms |
| `threshold` | 1.42 | Peak detection threshold on log-amplitude |
| `inside_radius_ang` | 90 | Resolution limit for center protection (Å) |
| `outside_radius_ang` | auto | Resolution limit for edge protection (Å) |
| `expand_pixel` | 10 | Morphological expansion of peak mask |
| `unit_cell_ang` | 116 | Crystal unit cell for shift calculation (Å) |

## Batch Processing

### Python API

```python
from lattice_subtraction import BatchProcessor, Config

config = Config(pixel_ang=0.56)
processor = BatchProcessor(config, num_workers=8)

# Process all MRC files in a directory
result = processor.process_directory(
    input_dir="raw_micrographs/",
    output_dir="subtracted/",
    pattern="*.mrc"
)

print(f"Processed {result.successful}/{result.total} files")
```

### Numbered Sequences

```python
# Process numbered file sequence (like legacy HYPER_loop scripts)
result = processor.process_numbered_sequence(
    input_pattern="data/mic_{num}.mrc",
    output_dir="processed/",
    start=1,
    end=1000,
    zero_pad=4  # mic_0001.mrc, mic_0002.mrc, ...
)
```

## GPU Acceleration

Enable GPU processing for 5-10× speedup on large images:

```python
config = Config(pixel_ang=0.56, backend="cupy")
subtractor = LatticeSubtractor(config)
```

Or via command line:

```bash
lattice-sub process input.mrc -o output.mrc --pixel-size 0.56 --gpu
```

## Migration from Legacy MATLAB Code

### Converting Configuration Files

```bash
# Convert legacy PARAMETER file to YAML
lattice-sub convert-config PARAMETER params.yaml
```

### API Correspondence

| MATLAB (Legacy) | Python (New) |
|-----------------|--------------|
| `LAsub('input.mrc')` | `subtractor.process("input.mrc")` |
| `bg_push_by_rot(...)` | `LatticeSubtractor._process_padded(...)` |
| `bg_FastSubtract_standard(pw)` | `processing.subtract_background(pw)` |
| `bg_drill_hole(box, r)` | `masks.create_circular_mask(shape, r)` |
| `ReadMRC(file)` | `io.read_mrc(file)` |
| `WriteMRC(data, 1, file)` | `io.write_mrc(data, file)` |

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{lattice_subtraction,
  title = {Lattice Subtraction for Cryo-EM Micrographs},
  author = {Kasinath Lab},
  year = {2026},
  url = {https://github.com/kasinath-lab/lattice-subtraction}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This package is a modernization of MATLAB code originally developed for nucleosome array imaging. The phase-preserving inpainting algorithm was designed to reveal protein complexes bound to 2D crystal surfaces.
