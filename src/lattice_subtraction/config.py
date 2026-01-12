"""
Configuration management for lattice subtraction.

This module handles loading, validation, and storage of processing parameters
from YAML configuration files or Python dictionaries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import yaml


@dataclass
class Config:
    """
    Configuration parameters for lattice subtraction processing.
    
    All resolution parameters are in Angstroms. The algorithm removes lattice
    peaks in the resolution range between inside_radius_ang and outside_radius_ang.
    
    Attributes:
        pixel_ang: Pixel size in Angstroms (detector-dependent, e.g., 0.56 for K3)
        inside_radius_ang: Inner resolution limit - FFT spots within this radius 
                          are preserved (low-frequency structural info). Default: 90Å
        outside_radius_ang: Outer resolution limit - spots beyond this are preserved.
                           If None, auto-calculated as pixel_ang * 2 + 0.2
        threshold: Peak detection threshold on log-amplitude FFT. Spots above this
                  value are identified as lattice peaks. Default: 1.42
        expand_pixel: Morphological expansion radius for mask dilation. Default: 10
        pad_origin_x: X padding offset in pixels. Default: 200
        pad_origin_y: Y padding offset in pixels. Default: 200 (use 1000 for K3)
        pad_output: If False, crop output to original size. Default: False
        unit_cell_ang: Crystal unit cell size in Angstroms for shift calculation.
                      Default: 116Å (nucleosome repeat)
        backend: Computation backend - 'numpy' for CPU, 'pytorch' for GPU. Default: 'numpy'
    """
    
    # Required parameter
    pixel_ang: float
    
    # Resolution limits
    inside_radius_ang: float = 90.0
    outside_radius_ang: Optional[float] = None  # Auto-calculated if None
    
    # Peak detection
    threshold: float = 1.42
    expand_pixel: int = 10
    
    # Padding
    pad_origin_x: int = 200
    pad_origin_y: int = 200
    pad_output: bool = False
    
    # Crystal parameters
    unit_cell_ang: float = 116.0  # Nucleosome repeat distance
    
    # Computation backend
    backend: Literal["numpy", "pytorch"] = "numpy"
    
    def __post_init__(self):
        """Validate and set auto-calculated parameters."""
        if self.pixel_ang <= 0:
            raise ValueError(f"pixel_ang must be positive, got {self.pixel_ang}")
        
        if self.inside_radius_ang <= 0:
            raise ValueError(f"inside_radius_ang must be positive, got {self.inside_radius_ang}")
            
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        
        # Auto-calculate outside radius if not provided
        if self.outside_radius_ang is None:
            self.outside_radius_ang = self.pixel_ang * 2 + 0.2
        
        if self.outside_radius_ang >= self.inside_radius_ang:
            raise ValueError(
                f"outside_radius_ang ({self.outside_radius_ang}) must be smaller than "
                f"inside_radius_ang ({self.inside_radius_ang})"
            )
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config instance with loaded parameters
            
        Example YAML format:
            pixel_ang: 0.56
            threshold: 1.56
            inside_radius_ang: 90
            # outside_radius_ang: auto  # Optional, auto-calculated if omitted
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle 'auto' string for outside_radius_ang
        if data.get('outside_radius_ang') == 'auto':
            data['outside_radius_ang'] = None
            
        return cls(**data)
    
    @classmethod
    def from_legacy_parameter_file(cls, path: str | Path) -> "Config":
        """
        Load configuration from legacy MATLAB PARAMETER file format.
        
        Args:
            path: Path to legacy PARAMETER file
            
        Returns:
            Config instance with loaded parameters
        """
        path = Path(path)
        params = {}
        
        # Mapping from legacy names to new names
        name_map = {
            'inside_radius_ang': 'inside_radius_ang',
            'outside_radius_ang': 'outside_radius_ang', 
            'pixel_ang': 'pixel_ang',
            'threshold': 'threshold',
        }
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                
                # Split on whitespace, handle comments with !
                parts = line.split('!')
                main_part = parts[0].strip()
                if not main_part:
                    continue
                    
                tokens = main_part.split()
                if len(tokens) >= 2:
                    name = tokens[0].lower()
                    try:
                        value = float(tokens[1])
                    except ValueError:
                        continue
                    
                    if name in name_map:
                        params[name_map[name]] = value
        
        return cls(**params)
    
    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Output path for YAML file
        """
        path = Path(path)
        
        data = {
            'pixel_ang': self.pixel_ang,
            'inside_radius_ang': self.inside_radius_ang,
            'outside_radius_ang': self.outside_radius_ang,
            'threshold': self.threshold,
            'expand_pixel': self.expand_pixel,
            'pad_origin_x': self.pad_origin_x,
            'pad_origin_y': self.pad_origin_y,
            'pad_output': self.pad_output,
            'unit_cell_ang': self.unit_cell_ang,
            'backend': self.backend,
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def copy(self, **updates) -> "Config":
        """
        Create a copy of this config with optional updates.
        
        Args:
            **updates: Parameters to override
            
        Returns:
            New Config instance with updates applied
        """
        from dataclasses import asdict
        current = asdict(self)
        current.update(updates)
        return Config(**current)


def create_default_config(pixel_ang: float = 0.56, detector: str = "K3") -> Config:
    """
    Create a config with detector-specific defaults.
    
    Args:
        pixel_ang: Pixel size in Angstroms
        detector: Detector type ('K3', 'Falcon', 'generic')
        
    Returns:
        Config with appropriate defaults for the detector
    """
    pad_y = 1000 if detector.upper() == "K3" else 200
    
    return Config(
        pixel_ang=pixel_ang,
        pad_origin_y=pad_y,
    )
