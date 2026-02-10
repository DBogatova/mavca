"""
Configuration management for the MAVCA Pipeline UI.

This module provides the PipelineConfig class for managing common parameters
(DATE, MOUSE, RUN) and base directory configuration.
"""

import re
from pathlib import Path
from typing import Dict, Any


class PipelineConfig:
    """Configuration for pipeline execution.
    
    Manages the common parameters (DATE, MOUSE, RUN) and base directory
    used across all pipeline scripts.
    
    Attributes:
        date: Date string in YYYY-MM-DD format
        mouse: Mouse identifier (alphanumeric, underscore, hyphen)
        run: Run identifier (alphanumeric, underscore, hyphen)
        base_dir: Base directory path for data storage
    """
    
    def __init__(self):
        """Initialize with default values."""
        self.date: str = ""
        self.mouse: str = ""
        self.run: str = ""
        self.base_dir: Path = (
            Path.home() / "Desktop" / "Boston_University" / 
            "Devor_Lab" / "apical-dendrites-2025" / "scape-data"
        )
    
    def validate_date(self, date_str: str) -> bool:
        """Validate DATE format (YYYY-MM-DD).
        
        Validates that the date string matches YYYY-MM-DD format with:
        - Year: 4 digits
        - Month: 01-12
        - Day: 01-31 (basic validation, doesn't check month-specific limits)
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid YYYY-MM-DD format, False otherwise
        """
        # Pattern: YYYY-MM-DD with valid month (01-12) and day (01-31)
        pattern = r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$'
        return bool(re.match(pattern, date_str))
    
    def validate_identifier(self, identifier: str) -> bool:
        """Validate MOUSE or RUN identifier.
        
        Validates that the identifier contains only:
        - Alphanumeric characters (a-z, A-Z, 0-9)
        - Underscores (_)
        - Hyphens (-)
        
        Args:
            identifier: Identifier string to validate
            
        Returns:
            True if valid identifier, False otherwise
        """
        # Must be non-empty and contain only alphanumeric, underscore, hyphen
        if not identifier:
            return False
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, identifier))
    
    def get_data_path(self) -> Path:
        """Construct DATE/MOUSE/RUN path.
        
        Constructs the full data path following the pattern:
        base_dir/DATE/MOUSE/RUN/
        
        Returns:
            Path object representing the data directory
        """
        return self.base_dir / self.date / self.mouse / self.run
    
    def data_path_exists(self) -> bool:
        """Check if data directory exists.
        
        Returns:
            True if the constructed data path exists, False otherwise
        """
        return self.get_data_path().exists()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "date": self.date,
            "mouse": self.mouse,
            "run": self.run,
            "base_dir": str(self.base_dir)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Deserialize from dictionary.
        
        Args:
            data: Dictionary containing configuration data
            
        Returns:
            PipelineConfig instance with values from dictionary
        """
        config = cls()
        config.date = data.get("date", "")
        config.mouse = data.get("mouse", "")
        config.run = data.get("run", "")
        if "base_dir" in data:
            config.base_dir = Path(data["base_dir"])
        return config
