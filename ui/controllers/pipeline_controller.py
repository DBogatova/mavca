"""
Main controller for the MAVCA Pipeline UI.

This module provides the PipelineController class that coordinates application
logic, state management, script execution, and module operations.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Callable

from ..models.pipeline_config import PipelineConfig
from ..models.progress_state import ProgressState
from ..models.state_manager import StateManager
from ..models.script_executor import ScriptExecutor
from ..models.pipeline_modules import MODULES, get_module
from ..models.module_definition import ModuleDefinition


class PipelineController:
    """Main controller for pipeline operations.
    
    Coordinates application logic and state management, providing a high-level
    interface for the UI to interact with the pipeline.
    
    Attributes:
        config: Current pipeline configuration
        state_manager: Handles state persistence
        executor: Handles script execution
        modules: Dictionary of all module definitions
        current_progress: Progress state for current dataset
    """
    
    def __init__(self):
        """Initialize the pipeline controller."""
        self.config = PipelineConfig()
        self.state_manager = StateManager(Path.home() / ".mavca_ui")
        self.executor = ScriptExecutor(self._find_venv())
        self.modules = MODULES
        self.current_progress: Optional[ProgressState] = None
    
    def initialize(self) -> None:
        """Initialize controller and load saved state.
        
        Loads the last saved configuration and corresponding progress state.
        If no saved configuration exists, uses default values.
        """
        # Load saved configuration
        saved_config = self.state_manager.load_config()
        if saved_config:
            self.config = saved_config
            
            # Load progress for the configured dataset
            if self.config.date and self.config.mouse and self.config.run:
                dataset_key = self._get_dataset_key()
                self.current_progress = self.state_manager.load_progress(dataset_key)
        
        # If no progress loaded, create empty progress
        if self.current_progress is None:
            dataset_key = self._get_dataset_key() if self.config.date else ""
            self.current_progress = ProgressState(dataset_key)
    
    def update_config(self, date: str, mouse: str, run: str) -> bool:
        """Update configuration and load corresponding progress.
        
        Validates the new configuration values, updates the config,
        saves it to disk, and loads the progress state for the new dataset.
        
        Args:
            date: Date string in YYYY-MM-DD format
            mouse: Mouse identifier
            run: Run identifier
            
        Returns:
            True if configuration is valid and updated, False otherwise
        """
        # Validate inputs
        if not self.config.validate_date(date):
            return False
        if not self.config.validate_identifier(mouse):
            return False
        if not self.config.validate_identifier(run):
            return False
        
        # Update configuration
        self.config.date = date
        self.config.mouse = mouse
        self.config.run = run
        
        # Save configuration
        try:
            self.state_manager.save_config(self.config)
        except IOError as e:
            print(f"Warning: Failed to save configuration: {e}")
        
        # Load progress for this dataset
        dataset_key = self._get_dataset_key()
        self.current_progress = self.state_manager.load_progress(dataset_key)
        
        return True
    
    def execute_module(
        self,
        module_id: str,
        output_callback: Callable[[str], None]
    ) -> str:
        """Execute all scripts for a module.
        
        Executes all non-optional scripts for the specified module sequentially.
        Updates progress state on successful completion. Raises exceptions on
        failure to allow the caller to handle errors appropriately.
        
        Args:
            module_id: Module identifier (e.g., "M1", "M2")
            output_callback: Callback function to receive output lines
            
        Returns:
            The module_id that was executed (for post-completion handling)
            
        Raises:
            KeyError: If module_id is not found
            RuntimeError: If a script is already running
            FileNotFoundError: If script or Python interpreter not found
            Exception: If script execution fails
        """
        # Get module definition
        module = get_module(module_id)
        
        # Execute each non-optional script
        for script_info in module.scripts:
            if script_info.optional:
                output_callback(f"Skipping optional script: {script_info.name}\n")
                continue
            
            output_callback(f"\n{'='*60}\n")
            output_callback(f"Executing: {script_info.name}\n")
            output_callback(f"Description: {script_info.description}\n")
            output_callback(f"{'='*60}\n\n")
            
            # Execute the script
            exit_code = self.executor.execute_script(
                script_info.path,
                self.config,
                output_callback
            )
            
            # Check exit code
            if exit_code != 0:
                error_msg = f"\nScript failed with exit code {exit_code}\n"
                output_callback(error_msg)
                raise Exception(f"Script execution failed: {script_info.name}")
            
            output_callback(f"\n✓ {script_info.name} completed successfully\n")
        
        # Mark module as complete
        if self.current_progress:
            self.current_progress.mark_complete(module_id)
            
            # Special handling for M2 - track if it was run with guides
            if module_id == "M2":
                # Check if M2.5 was completed (guides exist)
                if self.current_progress.is_complete("M2.5"):
                    self.current_progress.m2_guided = True
                else:
                    self.current_progress.m2_guided = False
            
            # Save progress
            try:
                self.state_manager.save_progress(self.current_progress)
            except IOError as e:
                output_callback(f"\nWarning: Failed to save progress: {e}\n")
        
        output_callback(f"\n{'='*60}\n")
        output_callback(f"Module {module_id} completed successfully!\n")
        output_callback(f"{'='*60}\n\n")
        
        # Return module_id to allow caller to handle post-completion actions
        return module_id
    
    def verify_outputs(self, module_id: str) -> Dict[str, bool]:
        """Check which expected outputs exist.
        
        Verifies the existence of all expected output files and directories
        for the specified module.
        
        Args:
            module_id: Module identifier (e.g., "M1", "M2")
            
        Returns:
            Dictionary mapping output paths to existence status (True/False)
            
        Raises:
            KeyError: If module_id is not found
        """
        module = get_module(module_id)
        data_path = self.config.get_data_path()
        
        output_status = {}
        for output_path in module.expected_outputs:
            full_path = data_path / output_path
            output_status[output_path] = full_path.exists()
        
        return output_status
    
    def launch_curation(self) -> None:
        """Launch M3 curation window.
        
        This method is a placeholder for launching the M3 curation interface.
        The actual implementation will be added when the CurationWindow view
        is implemented.
        
        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        # TODO: Implement M3 curation window launch
        # This will be implemented in task 16 when CurationWindow is created
        raise NotImplementedError(
            "M3 curation window is not yet implemented. "
            "This will be added in task 16."
        )
    
    def open_output_directory(self, module_id: str) -> None:
        """Open module output directory in file browser.
        
        Opens the data directory for the current dataset in the system's
        default file browser. Uses platform-specific commands with macOS
        integration for better path handling.
        
        Args:
            module_id: Module identifier (used for context, opens data dir)
            
        Raises:
            FileNotFoundError: If the data directory doesn't exist
        """
        from ..utils.macos_integration import MacOSIntegration
        
        data_path = self.config.get_data_path()
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {data_path}\n"
                f"Please check your DATE, MOUSE, and RUN configuration."
            )
        
        # Use macOS integration for better path handling
        MacOSIntegration.open_directory_native(data_path)
    
    def get_module(self, module_id: str) -> ModuleDefinition:
        """Get a module definition by ID.
        
        Args:
            module_id: Module identifier (e.g., "M1", "M2")
            
        Returns:
            ModuleDefinition for the specified module
            
        Raises:
            KeyError: If module_id is not found
        """
        return get_module(module_id)
    
    def is_module_complete(self, module_id: str) -> bool:
        """Check if a module is completed.
        
        Args:
            module_id: Module identifier to check
            
        Returns:
            True if module is completed, False otherwise
        """
        if self.current_progress:
            return self.current_progress.is_complete(module_id)
        return False
    
    def reset_progress(self) -> None:
        """Reset progress for the current dataset.
        
        Clears all completion status for the current dataset and saves
        the reset state to disk.
        """
        if self.current_progress:
            self.current_progress.reset()
            try:
                self.state_manager.save_progress(self.current_progress)
            except IOError as e:
                print(f"Warning: Failed to save progress: {e}")

    def check_labelmap_types(self) -> Dict[str, bool]:
        """Check which types of labelmaps exist.

        Returns:
            Dictionary with 'guided' and 'non_guided' keys indicating existence
        """
        data_path = self.config.get_data_path()
        labelmaps_dir = data_path / "labelmaps"

        result = {
            "guided": False,
            "non_guided": False
        }

        if not labelmaps_dir.exists():
            return result

        # Check for guided labelmaps (created after M2.5)
        # These typically have "guided" in the filename or are in a guided subdirectory
        guides_dir = data_path / "preprocessed" / "guides"
        if guides_dir.exists() and any(guides_dir.iterdir()):
            # If guides exist and M2 was run after M2.5, we have guided labelmaps
            if self.current_progress and self.current_progress.m2_guided:
                result["guided"] = True

        # Check for non-guided labelmaps
        # If M2 was completed but not with guides, we have non-guided
        if self.current_progress and self.current_progress.is_complete("M2"):
            if not self.current_progress.m2_guided:
                result["non_guided"] = True
            elif self.current_progress.is_complete("M2.5"):
                # If M2.5 exists, there might be both types
                result["non_guided"] = True

        return result
    
    def launch_curation(self) -> None:
        """Launch M3 curation window.
        
        Opens the interactive curation interface for reviewing and editing masks.
        
        Raises:
            FileNotFoundError: If labelmaps directory doesn't exist
        """
        from ..views.curation_window import CurationWindow
        
        data_path = self.config.get_data_path()
        labelmaps_dir = data_path / "labelmaps"
        
        if not labelmaps_dir.exists():
            raise FileNotFoundError(
                f"Labelmaps directory not found: {labelmaps_dir}\n"
                "Please run M2 first to generate masks."
            )
        
        # Create and show curation window
        curation_window = CurationWindow(data_path)
        curation_window.show()
        
        # Note: Window is non-modal, so it doesn't block the main UI
    
    def _get_dataset_key(self) -> str:
        """Get the dataset key for the current configuration.
        
        Returns:
            Dataset key in DATE_MOUSE_RUN format
        """
        return f"{self.config.date}_{self.config.mouse}_{self.config.run}"
    
    def _find_venv(self) -> Path:
        """Locate the project's virtual environment.
        
        Searches for common virtual environment directory names in the
        project root and parent directories.
        
        Returns:
            Path to the virtual environment directory
            
        Raises:
            FileNotFoundError: If no virtual environment is found
        """
        # Start from the current file's location and work up to project root
        current_dir = Path(__file__).resolve().parent
        
        # Go up to project root (ui/controllers -> ui -> project_root)
        project_root = current_dir.parent.parent
        
        # Common virtual environment names
        venv_names = [".venv311", ".venv", "venv", "env"]
        
        # Search in project root
        for venv_name in venv_names:
            venv_path = project_root / venv_name
            if venv_path.exists() and venv_path.is_dir():
                return venv_path
        
        # If not found, raise an error
        raise FileNotFoundError(
            f"Could not find virtual environment in {project_root}. "
            f"Searched for: {', '.join(venv_names)}"
        )
