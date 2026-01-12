"""
Tests for lattice subtraction package.
"""

import numpy as np
import pytest
from pathlib import Path


class TestConfig:
    """Tests for configuration handling."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        from lattice_subtraction.config import Config
        
        config = Config(pixel_ang=0.56)
        assert config.pixel_ang == 0.56
        assert config.threshold == 1.42
        assert config.inside_radius_ang == 90.0
        # Auto-calculated outside radius
        assert config.outside_radius_ang == pytest.approx(0.56 * 2 + 0.2)
    
    def test_config_validation(self):
        """Test config validation."""
        from lattice_subtraction.config import Config
        
        with pytest.raises(ValueError, match="pixel_ang must be positive"):
            Config(pixel_ang=-1.0)
        
        with pytest.raises(ValueError, match="outside_radius_ang"):
            Config(pixel_ang=0.56, outside_radius_ang=100)  # > inside_radius
    
    def test_config_yaml_roundtrip(self, tmp_path):
        """Test saving and loading config from YAML."""
        from lattice_subtraction.config import Config
        
        config = Config(pixel_ang=0.56, threshold=1.56)
        yaml_path = tmp_path / "config.yaml"
        
        config.to_yaml(yaml_path)
        loaded = Config.from_yaml(yaml_path)
        
        assert loaded.pixel_ang == config.pixel_ang
        assert loaded.threshold == config.threshold


class TestMasks:
    """Tests for mask generation."""
    
    def test_circular_mask(self):
        """Test circular mask creation."""
        from lattice_subtraction.masks import create_circular_mask
        
        mask = create_circular_mask((100, 100), radius=30)
        
        assert mask.shape == (100, 100)
        assert mask[50, 50]  # Center should be True
        assert not mask[0, 0]  # Corner should be False
    
    def test_radial_band_mask(self):
        """Test annular mask creation."""
        from lattice_subtraction.masks import create_radial_band_mask
        
        mask = create_radial_band_mask((100, 100), inner_radius=20, outer_radius=40)
        
        assert not mask[50, 50]  # Center should be False (inside inner)
        assert mask[50, 75]  # Should be True (in band)
        assert not mask[0, 0]  # Corner should be False (outside outer)
    
    def test_resolution_to_pixels(self):
        """Test resolution to pixel conversion."""
        from lattice_subtraction.masks import resolution_to_pixels
        
        # At 1 Å/pixel, 10 Å resolution in 100 pixel box = 10 pixel radius
        result = resolution_to_pixels(
            resolution_ang=10,
            pixel_size_ang=1.0,
            box_size=100,
        )
        assert result == pytest.approx(10.0)


class TestProcessing:
    """Tests for image processing utilities."""
    
    def test_pad_image(self):
        """Test image padding."""
        from lattice_subtraction.processing import pad_image, crop_to_original
        
        img = np.random.randn(100, 80).astype(np.float32)
        
        padded, meta = pad_image(img, pad_origin=(10, 10))
        
        assert padded.shape[0] > img.shape[0]
        assert padded.shape[1] > img.shape[1]
        
        # Test crop back
        cropped = crop_to_original(padded, meta)
        np.testing.assert_array_equal(cropped.shape, img.shape)
    
    def test_subtract_background(self):
        """Test background subtraction."""
        from lattice_subtraction.processing import subtract_background
        
        # Create image with gradient background + peaks
        y, x = np.ogrid[:100, :100]
        background = (x + y).astype(np.float32) / 200
        peaks = np.zeros((100, 100), dtype=np.float32)
        peaks[50, 50] = 10
        peaks[30, 70] = 10
        
        img = background + peaks
        
        result = subtract_background(img)
        
        # Peaks should be more prominent after subtraction
        assert result[50, 50] > result[10, 10]
    
    def test_shift_and_average(self):
        """Test shift averaging for inpainting."""
        from lattice_subtraction.processing import shift_and_average
        
        img = np.ones((100, 100), dtype=np.float32)
        img[50, 50] = 10  # Single peak
        
        result = shift_and_average(img, shift_pixels=5)
        
        # Peak should be smoothed out
        assert result[50, 50] < img[50, 50]


class TestCore:
    """Tests for core lattice subtraction."""
    
    def test_subtractor_creation(self):
        """Test LatticeSubtractor initialization."""
        from lattice_subtraction.core import LatticeSubtractor
        from lattice_subtraction.config import Config
        
        config = Config(pixel_ang=0.56)
        subtractor = LatticeSubtractor(config)
        
        assert subtractor.config == config
        assert not subtractor.use_gpu
    
    def test_process_array(self):
        """Test processing a numpy array directly."""
        from lattice_subtraction.core import LatticeSubtractor
        from lattice_subtraction.config import Config
        
        config = Config(pixel_ang=1.0, threshold=2.0)
        subtractor = LatticeSubtractor(config)
        
        # Create synthetic image with lattice pattern
        img = np.random.randn(256, 256).astype(np.float32)
        
        # Add periodic pattern
        y, x = np.ogrid[:256, :256]
        lattice = np.sin(2 * np.pi * x / 20) + np.sin(2 * np.pi * y / 20)
        img += lattice.astype(np.float32) * 5
        
        result = subtractor.process_array(img)
        
        assert result.image.shape == img.shape
        assert result.original_shape == img.shape


class TestIO:
    """Tests for MRC I/O (requires test data)."""
    
    def test_mrc_roundtrip(self, tmp_path):
        """Test writing and reading MRC file."""
        from lattice_subtraction.io import read_mrc, write_mrc
        
        # Create test data
        data = np.random.randn(100, 100).astype(np.float32)
        
        mrc_path = tmp_path / "test.mrc"
        write_mrc(data, mrc_path)
        
        loaded = read_mrc(mrc_path)
        
        np.testing.assert_array_almost_equal(data, loaded, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
