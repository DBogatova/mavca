"""
State management for the MAVCA Pipeline UI.

This module provides the StateManager class for persisting configuration
and progress state to JSON files.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from .pipeline_config import PipelineConfig
from .progress_state import ProgressState


class StateManager:
    """Manages application state persistence.
    
    Handles saving and loading of configuration and progress state to/from
    JSON files. Provides graceful handling of corrupted or missing files.
    
    Attributes:
        state_dir: Directory for storing state files
        config_file: Path to configuration JSON file
        progress_file: Path to progress JSON file
    """
    
    def __init__(self, state_dir: Path):
        """Initialize state manager.
        
        Args:
            state_dir: Directory path for storing state files
        """
        self.state_dir: Path = Path(state_dir)
        self.config_file: Path = self.state_dir / "config.json"
        self.progress_file: Path = self.state_dir / "progress.json"
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: PipelineConfig) -> None:
        """Save configuration to disk.
        
        Serializes the configuration to JSON and writes to the config file.
        
        Args:
            config: PipelineConfig instance to save
            
        Raises:
            IOError: If unable to write to config file
        """
        try:
            config_data = config.to_dict()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
        except (IOError, OSError) as e:
            raise IOError(f"Failed to save configuration: {e}") from e
    
    def load_config(self) -> Optional[PipelineConfig]:
        """Load configuration from disk.
        
        Reads and deserializes configuration from the config file.
        Returns None if file doesn't exist or is corrupted.
        
        Returns:
            PipelineConfig instance if successful, None otherwise
        """
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return PipelineConfig.from_dict(config_data)
        except (json.JSONDecodeError, IOError, OSError, KeyError) as e:
            # Log error but don't crash - return None to use defaults
            print(f"Warning: Failed to load configuration: {e}")
            return None
    
    def save_progress(self, progress: ProgressState) -> None:
        """Save progress state to disk.
        
        Loads existing progress data, updates with the given progress state,
        and writes back to the progress file. This allows multiple datasets
        to have their progress tracked in a single file.
        
        Args:
            progress: ProgressState instance to save
            
        Raises:
            IOError: If unable to write to progress file
        """
        try:
            # Load existing progress data
            all_progress = self._load_all_progress()
            
            # Update with new progress
            all_progress[progress.dataset_key] = progress.to_dict()
            
            # Write back to file
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(all_progress, f, indent=2)
        except (IOError, OSError) as e:
            raise IOError(f"Failed to save progress: {e}") from e
    
    def load_progress(self, dataset_key: str) -> ProgressState:
        """Load progress state for a dataset.
        
        Reads progress data from file and returns the progress state for
        the specified dataset. If no progress exists for the dataset,
        returns a new empty progress state.
        
        Args:
            dataset_key: Dataset identifier (DATE_MOUSE_RUN format)
            
        Returns:
            ProgressState instance for the dataset
        """
        all_progress = self._load_all_progress()
        
        if dataset_key in all_progress:
            try:
                return ProgressState.from_dict(all_progress[dataset_key])
            except (KeyError, ValueError) as e:
                # Corrupted progress data for this dataset - return new state
                print(f"Warning: Corrupted progress data for {dataset_key}: {e}")
                return ProgressState(dataset_key)
        else:
            # No progress for this dataset yet
            return ProgressState(dataset_key)
    
    def list_datasets(self) -> List[str]:
        """List all datasets with saved progress.
        
        Returns:
            List of dataset keys (DATE_MOUSE_RUN format)
        """
        all_progress = self._load_all_progress()
        return list(all_progress.keys())
    
    def _load_all_progress(self) -> Dict[str, Any]:
        """Load all progress data from file.
        
        Internal helper method to load the complete progress file.
        Returns empty dict if file doesn't exist or is corrupted.
        
        Returns:
            Dictionary mapping dataset keys to progress data
        """
        if not self.progress_file.exists():
            return {}
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            # Corrupted progress file - log and return empty dict
            print(f"Warning: Failed to load progress file: {e}")
            return {}
