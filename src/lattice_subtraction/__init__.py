"""
Lattice Subtraction for Cryo-EM Micrographs

This package provides tools for computationally removing periodic crystal lattice
signals (Bragg spots) from cryo-EM micrographs to reveal non-periodic features
such as defects, individual particles, or molecular tags.

Main components:
    - LatticeSubtractor: Core class for processing single images
    - BatchProcessor: Parallel processing of multiple micrographs
    - Config: Configuration management via YAML files
    - generate_visualizations: Create comparison PNG images

Example:
    >>> from lattice_subtraction import LatticeSubtractor, Config
    >>> config = Config.from_yaml("params.yaml")
    >>> subtractor = LatticeSubtractor(config)
    >>> result = subtractor.process("micrograph.mrc")
    >>> result.save("output.mrc")
"""

__version__ = "1.2.0"
__author__ = "George Stephenson & Vignesh Kasinath"

from .config import Config
from .core import LatticeSubtractor
from .batch import BatchProcessor
from .io import read_mrc, write_mrc
from .threshold_optimizer import (
    ThresholdOptimizer,
    OptimizationResult,
    find_optimal_threshold,
)
from .visualization import (
    generate_visualizations,
    save_comparison_visualization,
    create_comparison_figure,
)
from .processing import subtract_background_gpu
from .ui import TerminalUI, get_ui, is_interactive

__all__ = [
    "LatticeSubtractor",
    "BatchProcessor", 
    "Config",
    "read_mrc",
    "write_mrc",
    "generate_visualizations",
    "save_comparison_visualization",
    "create_comparison_figure",
    "TerminalUI",
    "get_ui",
    "is_interactive",
    "ThresholdOptimizer",
    "OptimizationResult",
    "find_optimal_threshold",
    "subtract_background_gpu",
    "__version__",
]
