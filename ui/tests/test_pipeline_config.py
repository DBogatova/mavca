"""
Unit tests for PipelineConfig class.

Tests configuration management, validation, path construction,
and serialization functionality.
"""

import pytest
from pathlib import Path
from ui.models.pipeline_config import PipelineConfig


class TestPipelineConfig:
    """Test suite for PipelineConfig class."""
    
    def test_initialization(self):
        """Test default initialization values."""
        config = PipelineConfig()
        assert config.date == ""
        assert config.mouse == ""
        assert config.run == ""
        assert isinstance(config.base_dir, Path)
    
    def test_validate_date_valid_formats(self):
        """Test date validation with valid YYYY-MM-DD formats."""
        config = PipelineConfig()
        
        # Valid dates
        assert config.validate_date("2025-01-15") is True
        assert config.validate_date("2024-12-31") is True
        assert config.validate_date("2023-06-01") is True
        assert config.validate_date("2025-12-25") is True
    
    def test_validate_date_invalid_formats(self):
        """Test date validation with invalid formats."""
        config = PipelineConfig()
        
        # Invalid formats
        assert config.validate_date("2025-1-15") is False  # Single digit month
        assert config.validate_date("2025-01-5") is False  # Single digit day
        assert config.validate_date("25-01-15") is False  # Two digit year
        assert config.validate_date("2025/01/15") is False  # Wrong separator
        assert config.validate_date("2025-13-01") is False  # Invalid month
        assert config.validate_date("2025-00-01") is False  # Invalid month
        assert config.validate_date("2025-01-32") is False  # Invalid day
        assert config.validate_date("2025-01-00") is False  # Invalid day
        assert config.validate_date("") is False  # Empty string
        assert config.validate_date("not-a-date") is False  # Random text
    
    def test_validate_identifier_valid(self):
        """Test identifier validation with valid inputs."""
        config = PipelineConfig()
        
        # Valid identifiers
        assert config.validate_identifier("rAi162_phpeb") is True
        assert config.validate_identifier("run1") is True
        assert config.validate_identifier("run4-crop") is True
        assert config.validate_identifier("organoid") is True
        assert config.validate_identifier("test_123") is True
        assert config.validate_identifier("ABC-123_xyz") is True
    
    def test_validate_identifier_invalid(self):
        """Test identifier validation with invalid inputs."""
        config = PipelineConfig()
        
        # Invalid identifiers
        assert config.validate_identifier("") is False  # Empty string
        assert config.validate_identifier("run 1") is False  # Space
        assert config.validate_identifier("run.1") is False  # Period
        assert config.validate_identifier("run@1") is False  # Special char
        assert config.validate_identifier("run/1") is False  # Slash
        assert config.validate_identifier("run\\1") is False  # Backslash
    
    def test_get_data_path(self):
        """Test data path construction."""
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "rAi162_phpeb"
        config.run = "run1"
        
        expected_path = config.base_dir / "data" / "2025-12-25" / "rAi162_phpeb" / "run1"
        assert config.get_data_path() == expected_path
    
    def test_get_data_path_with_custom_base_dir(self):
        """Test data path construction with custom base directory."""
        config = PipelineConfig()
        config.base_dir = Path("/custom/path")
        config.date = "2025-01-01"
        config.mouse = "test_mouse"
        config.run = "test_run"
        
        expected_path = Path("/custom/path/data/2025-01-01/test_mouse/test_run")
        assert config.get_data_path() == expected_path
    
    def test_data_path_exists(self, tmp_path):
        """Test data path existence checking."""
        config = PipelineConfig()
        config.base_dir = tmp_path
        config.date = "2025-01-01"
        config.mouse = "test_mouse"
        config.run = "test_run"
        
        # Path doesn't exist yet
        assert config.data_path_exists() is False
        
        # Create the path
        data_path = config.get_data_path()
        data_path.mkdir(parents=True)
        
        # Now it should exist
        assert config.data_path_exists() is True
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "rAi162_phpeb"
        config.run = "run1"
        config.base_dir = Path("/test/path")
        
        result = config.to_dict()
        
        assert result["date"] == "2025-12-25"
        assert result["mouse"] == "rAi162_phpeb"
        assert result["run"] == "run1"
        assert result["base_dir"] == "/test/path"
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "date": "2025-12-25",
            "mouse": "rAi162_phpeb",
            "run": "run1",
            "base_dir": "/test/path"
        }
        
        config = PipelineConfig.from_dict(data)
        
        assert config.date == "2025-12-25"
        assert config.mouse == "rAi162_phpeb"
        assert config.run == "run1"
        assert config.base_dir == Path("/test/path")
    
    def test_from_dict_with_missing_fields(self):
        """Test deserialization with missing fields uses defaults."""
        data = {
            "date": "2025-12-25"
        }
        
        config = PipelineConfig.from_dict(data)
        
        assert config.date == "2025-12-25"
        assert config.mouse == ""
        assert config.run == ""
        assert isinstance(config.base_dir, Path)
    
    def test_serialization_round_trip(self):
        """Test that serialization and deserialization preserve values."""
        original = PipelineConfig()
        original.date = "2025-12-25"
        original.mouse = "rAi162_phpeb"
        original.run = "run1"
        original.base_dir = Path("/test/path")
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = PipelineConfig.from_dict(data)
        
        # Verify all values match
        assert restored.date == original.date
        assert restored.mouse == original.mouse
        assert restored.run == original.run
        assert restored.base_dir == original.base_dir
