"""
Progress tracking for the MAVCA Pipeline UI.

This module provides the ProgressState class for tracking which modules
have been completed for a specific dataset.
"""

from datetime import datetime
from typing import Set, Dict, Any


class ProgressState:
    """Progress tracking for a specific dataset.
    
    Tracks which modules have been completed and whether M2 was run
    with or without guides.
    
    Attributes:
        dataset_key: Unique identifier for the dataset (DATE_MOUSE_RUN)
        completed_modules: Set of completed module IDs
        m2_guided: Whether M2 was run with guides from M2.5
        last_updated: Timestamp of last update
    """
    
    def __init__(self, dataset_key: str):
        """Initialize progress state for a dataset.
        
        Args:
            dataset_key: Unique identifier (DATE_MOUSE_RUN format)
        """
        self.dataset_key: str = dataset_key
        self.completed_modules: Set[str] = set()
        self.m2_guided: bool = False
        self.last_updated: datetime = datetime.now()
    
    def mark_complete(self, module_id: str) -> None:
        """Mark a module as completed.
        
        Args:
            module_id: Module identifier (e.g., "M1", "M2", "M3")
        """
        self.completed_modules.add(module_id)
        self.last_updated = datetime.now()
    
    def is_complete(self, module_id: str) -> bool:
        """Check if a module is completed.
        
        Args:
            module_id: Module identifier to check
            
        Returns:
            True if module is completed, False otherwise
        """
        return module_id in self.completed_modules
    
    def reset(self) -> None:
        """Reset all progress."""
        self.completed_modules.clear()
        self.m2_guided = False
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.
        
        Returns:
            Dictionary representation of progress state
        """
        return {
            "dataset_key": self.dataset_key,
            "completed_modules": list(self.completed_modules),
            "m2_guided": self.m2_guided,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressState':
        """Deserialize from dictionary.
        
        Args:
            data: Dictionary containing progress state data
            
        Returns:
            ProgressState instance with values from dictionary
        """
        dataset_key = data.get("dataset_key", "")
        progress = cls(dataset_key)
        progress.completed_modules = set(data.get("completed_modules", []))
        progress.m2_guided = data.get("m2_guided", False)
        
        # Parse last_updated timestamp
        if "last_updated" in data:
            try:
                progress.last_updated = datetime.fromisoformat(data["last_updated"])
            except (ValueError, TypeError):
                progress.last_updated = datetime.now()
        
        return progress
