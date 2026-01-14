# Lattice Subtraction for Cryo-EM

```
.__          __    __  .__                                   ___.    
|  | _____ _/  |__/  |_|__| ____  ____             ________ _\_ |__  
|  | \__  \\   __\   __\  |/ ___\/ __ \   ______  /  ___/  |  \ __ \ 
|  |__/ __ \|  |  |  | |  \  \__\  ___/  /_____/  \___ \|  |  / \_\ \
|____(____  /__|  |__| |__|\___  >___  >         /____  >____/|___  /
          \/                   \/    \/               \/          \/ 
```

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Remove periodic lattice patterns from cryo-EM micrographs to reveal non-periodic features.**

![Example Result](docs/images/example_comparison.png)

---

## Installation

```bash
pip install lattice-sub
```

That's it! GPU acceleration works automatically if you have an NVIDIA GPU.

---

## Quick Start

### Process a Single Image

```bash
lattice-sub process your_image.mrc -o output.mrc --pixel-size 0.56
```

### Process a Folder of Images

```bash
lattice-sub batch input_folder/ output_folder/ --pixel-size 0.56
```

### Generate Comparison Images

```bash
lattice-sub batch input_folder/ output_folder/ --pixel-size 0.56 --vis comparisons/
```

This creates side-by-side PNG images showing before/after/difference for each micrograph.

---

## Common Options

| Option | Description |
|--------|-------------|
| `-p, --pixel-size` | **Required.** Pixel size in Ångstroms |
| `-o, --output` | Output file path (default: `sub_<input>`) |
| `-t, --threshold` | Peak detection sensitivity (default: 1.42) |
| `--cpu` | Force CPU processing (GPU is used by default) |
| `-q, --quiet` | Hide the banner and progress messages |
| `-v, --verbose` | Show detailed processing information |

### Example with Options

```bash
# Process with custom threshold, verbose output
lattice-sub process image.mrc -o cleaned.mrc -p 0.56 -t 1.5 -v

# Batch process, force CPU with 8 parallel workers
lattice-sub batch raw/ processed/ -p 0.56 --cpu -j 8
```

---

## Using a Config File

For reproducible processing, save your parameters in a YAML file:

```yaml
# params.yaml
pixel_ang: 0.56
threshold: 1.42
inside_radius_ang: 90
unit_cell_ang: 116
```

Then use it:

```bash
lattice-sub process image.mrc -o output.mrc --config params.yaml
```

Create a starter config file:

```bash
lattice-sub init-config params.yaml --pixel-size 0.56
```

---

## GPU Troubleshooting

GPU should work automatically. If it doesn't:

```bash
# Check GPU status
lattice-sub setup-gpu

# Or verify manually
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

**Requirements:** NVIDIA GPU with driver 525+ (RTX 20/30/40 series, A100, etc.)

---

## Python API

```python
from lattice_subtraction import LatticeSubtractor, Config

# Configure and process
config = Config(pixel_ang=0.56)
subtractor = LatticeSubtractor(config)

result = subtractor.process("input.mrc")
result.save("output.mrc")
```

### Batch Processing

```python
from lattice_subtraction import BatchProcessor, Config

config = Config(pixel_ang=0.56)
processor = BatchProcessor(config)

stats = processor.process_directory("raw/", "processed/")
print(f"Processed {stats.successful}/{stats.total} files")
```

---

## What It Does

This tool removes the periodic "lattice" pattern from 2D crystal cryo-EM images:

1. **Finds lattice peaks** in the Fourier transform (the repeating pattern)
2. **Replaces them** with averaged local values (inpainting)
3. **Preserves phase** so the image stays accurate
4. **Returns** the cleaned image with non-periodic features visible

### Key Parameters Explained

| Parameter | What it controls |
|-----------|------------------|
| `pixel_ang` | Your detector's pixel size (check your microscope settings) |
| `threshold` | How aggressively to detect peaks. Higher = fewer peaks removed |
| `inside_radius_ang` | Protect low-resolution features (default 90Å is good for nucleosomes) |
| `unit_cell_ang` | Crystal repeat distance (116Å for nucleosome arrays) |

---

## Need Help?

```bash
# See all commands
lattice-sub --help

# See options for a specific command
lattice-sub process --help
lattice-sub batch --help
```

---

## Citation

```bibtex
@software{lattice_sub,
  title = {Lattice Subtraction for Cryo-EM Micrographs},
  author = {Stephenson, George and Kasinath, Vignesh},
  year = {2026},
  url = {https://github.com/gsstephenson/cryoem-lattice-subtraction}
}
```

**Original MATLAB algorithm**: Vignesh Kasinath  
**Python package**: George Stephenson  
Kasinath Lab, BioFrontiers Institute, University of Colorado Boulder

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<details>
<summary><strong>📚 Advanced Topics</strong> (click to expand)</summary>

### Algorithm Details

```
Image → Pad → FFT → Detect Peaks → Create Mask → Inpaint → iFFT → Crop → Output
```

The algorithm:
1. Pads the image to reduce edge artifacts
2. Applies 2D FFT to get frequency domain
3. Detects lattice peaks via thresholding on log-power spectrum
4. Creates a mask protecting inner (low-freq) and outer (high-freq) regions
5. Inpaints peaks with local average amplitude from 4 shifted neighbors
6. Preserves original phase information
7. Inverse FFT returns to real space

### Full Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pixel_ang` | *required* | Pixel size in Ångstroms |
| `threshold` | 1.42 | Peak detection threshold on log-amplitude |
| `inside_radius_ang` | 90 | Inner resolution limit (Å) - protects structural info |
| `outside_radius_ang` | auto | Outer resolution limit (Å) - protects near Nyquist |
| `expand_pixel` | 10 | Morphological expansion of peak mask (pixels) |
| `unit_cell_ang` | 116 | Crystal unit cell for inpaint shift calculation (Å) |
| `backend` | auto | `"auto"`, `"numpy"` (CPU), or `"pytorch"` (GPU) |

### Supported Hardware

- **NVIDIA GPUs**: RTX 20/30/40 series, A100, H100
- **CUDA versions**: 11.8, 12.1, 12.4, 12.6, 12.8
- **CPU fallback**: Works on any system (just slower)

### Development Setup

```bash
git clone https://github.com/gsstephenson/cryoem-lattice-subtraction.git
cd cryoem-lattice-subtraction

conda create -n lattice_sub python=3.11 -y
conda activate lattice_sub

pip install -e ".[dev]"
pytest tests/ -v  # Run tests
```

### Code Structure

```
src/lattice_subtraction/
├── __init__.py        # Package exports
├── cli.py             # Command-line interface
├── core.py            # LatticeSubtractor main class
├── batch.py           # Parallel batch processing
├── config.py          # Configuration dataclass
├── io.py              # MRC file I/O
├── masks.py           # FFT mask generation
├── processing.py      # FFT helpers
├── ui.py              # Terminal UI
└── visualization.py   # Comparison figures
```

### Migration from MATLAB

This is a modernized rewrite of legacy MATLAB code (`LAsub.m`, `bg_push_by_rot.m`, etc.) with:
- 10-100× faster processing via GPU
- Pip-installable package
- YAML config files instead of hardcoded values
- Command-line interface
- Automated testing

Convert a legacy PARAMETER file:

```bash
lattice-sub convert-config PARAMETER params.yaml
```

</details>
