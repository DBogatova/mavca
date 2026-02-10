"""
User-friendly error messages for the MAVCA Pipeline UI.

This module provides standardized error messages and error handling utilities.
"""

from pathlib import Path
from typing import Optional


class ErrorMessages:
    """Collection of user-friendly error messages."""
    
    # Configuration errors
    INVALID_DATE = (
        "Date must be in YYYY-MM-DD format (e.g., 2025-12-25).\n\n"
        "Please enter a valid date with:\n"
        "• Four-digit year (YYYY)\n"
        "• Two-digit month (01-12)\n"
        "• Two-digit day (01-31)"
    )
    
    INVALID_IDENTIFIER = (
        "Must contain only letters, numbers, underscores, and hyphens.\n\n"
        "Valid examples:\n"
        "• rAi162_phpeb\n"
        "• run1\n"
        "• organoid-2025\n\n"
        "Invalid examples:\n"
        "• run 1 (contains space)\n"
        "• test@run (contains @)\n"
        "• data/run (contains /)"
    )
    
    MISSING_DATA_DIR = (
        "Data directory does not exist:\n{path}\n\n"
        "Please check your DATE, MOUSE, and RUN values.\n\n"
        "The expected directory structure is:\n"
        "data/DATE/MOUSE/RUN/\n\n"
        "Suggestions:\n"
        "• Verify the DATE, MOUSE, and RUN parameters are correct\n"
        "• Check that the base data directory is set correctly\n"
        "• Ensure the directory has been created on the filesystem"
    )
    
    MISSING_BASE_DIR = (
        "Base data directory does not exist:\n{path}\n\n"
        "Please configure the correct base directory path.\n\n"
        "The base directory should contain the 'data' folder with your datasets."
    )
    
    # Execution errors
    MISSING_VENV = (
        "Could not find Python virtual environment.\n\n"
        "Please ensure one of the following exists in the project directory:\n"
        "• .venv311\n"
        "• .venv\n"
        "• venv\n"
        "• env\n\n"
        "To create a virtual environment:\n"
        "python3.11 -m venv .venv311\n"
        "source .venv311/bin/activate\n"
        "pip install -r requirements.txt"
    )
    
    SCRIPT_NOT_FOUND = (
        "Script file not found:\n{path}\n\n"
        "The script may have been moved or deleted.\n\n"
        "Suggestions:\n"
        "• Verify the project structure is intact\n"
        "• Check that all pipeline scripts are in the correct locations\n"
        "• Re-clone the repository if files are missing"
    )
    
    SCRIPT_FAILED = (
        "Script failed with exit code {code}.\n\n"
        "Check the console output for details about what went wrong.\n\n"
        "Common causes:\n"
        "• Missing input files from previous modules\n"
        "• Incorrect parameter values\n"
        "• Insufficient memory or disk space\n"
        "• Data format issues"
    )
    
    SCRIPT_TIMEOUT = (
        "Script execution timed out after {timeout} seconds.\n\n"
        "The script may be stuck or taking longer than expected.\n\n"
        "Suggestions:\n"
        "• Check the console output for the last operation\n"
        "• Verify the input data is not corrupted\n"
        "• Consider running the script manually to debug"
    )
    
    EXECUTION_IN_PROGRESS = (
        "A script is already running.\n\n"
        "Please wait for the current execution to complete before starting another.\n\n"
        "You can stop the current execution using the Stop button."
    )
    
    # Data errors
    MISSING_INPUTS = (
        "Required input files are missing:\n{files}\n\n"
        "Please run previous modules first to generate the required inputs.\n\n"
        "Workflow order:\n"
        "M1 → M1.5 (optional) → M2 → M2.5 (optional) → M2 (if M2.5 run) → M3 → M4 → M5 → M6"
    )
    
    MISSING_OUTPUTS = (
        "Expected outputs were not created:\n{files}\n\n"
        "The script may have failed or encountered an error.\n\n"
        "Suggestions:\n"
        "• Check the console output for error messages\n"
        "• Verify input files exist and are not corrupted\n"
        "• Try running the module again\n"
        "• Check disk space availability"
    )
    
    CORRUPTED_DATA = (
        "Data file appears to be corrupted:\n{path}\n\n"
        "The file may be incomplete or damaged.\n\n"
        "Suggestions:\n"
        "• Re-run the module that created this file\n"
        "• Check disk space and file permissions\n"
        "• Verify the source data is intact"
    )
    
    # State errors
    CORRUPTED_STATE = (
        "Application state file is corrupted:\n{path}\n\n"
        "The application will continue with default settings.\n\n"
        "Your progress data may need to be manually recovered from:\n"
        "{state_dir}"
    )
    
    PERMISSION_ERROR = (
        "Permission denied when accessing:\n{path}\n\n"
        "Please check file permissions.\n\n"
        "You may need to:\n"
        "• Change file permissions (chmod)\n"
        "• Run the application with appropriate privileges\n"
        "• Move the data to a location you have write access to"
    )
    
    # General errors
    UNEXPECTED_ERROR = (
        "An unexpected error occurred:\n{error}\n\n"
        "Please check the log file for details:\n"
        "{log_path}\n\n"
        "If the problem persists, please report this issue with the log file."
    )
    
    @staticmethod
    def format_missing_data_dir(path: Path) -> str:
        """Format missing data directory error message.
        
        Args:
            path: Path to the missing directory
            
        Returns:
            Formatted error message
        """
        return ErrorMessages.MISSING_DATA_DIR.format(path=path)
    
    @staticmethod
    def format_missing_base_dir(path: Path) -> str:
        """Format missing base directory error message.
        
        Args:
            path: Path to the missing base directory
            
        Returns:
            Formatted error message
        """
        return ErrorMessages.MISSING_BASE_DIR.format(path=path)
    
    @staticmethod
    def format_script_not_found(path: Path) -> str:
        """Format script not found error message.
        
        Args:
            path: Path to the missing script
            
        Returns:
            Formatted error message
        """
        return ErrorMessages.SCRIPT_NOT_FOUND.format(path=path)
    
    @staticmethod
    def format_script_failed(code: int) -> str:
        """Format script failed error message.
        
        Args:
            code: Exit code from the script
            
        Returns:
            Formatted error message
        """
        return ErrorMessages.SCRIPT_FAILED.format(code=code)
    
    @staticmethod
    def format_script_timeout(timeout: int) -> str:
        """Format script timeout error message.
        
        Args:
            timeout: Timeout duration in seconds
            
        Returns:
            Formatted error message
        """
        return ErrorMessages.SCRIPT_TIMEOUT.format(timeout=timeout)
    
    @staticmethod
    def format_missing_inputs(files: list) -> str:
        """Format missing inputs error message.
        
        Args:
            files: List of missing file paths
            
        Returns:
            Formatted error message
        """
        files_str = "\n".join(f"• {f}" for f in files)
        return ErrorMessages.MISSING_INPUTS.format(files=files_str)
    
    @staticmethod
    def format_missing_outputs(files: list) -> str:
        """Format missing outputs error message.
        
        Args:
            files: List of missing output paths
            
        Returns:
            Formatted error message
        """
        files_str = "\n".join(f"• {f}" for f in files)
        return ErrorMessages.MISSING_OUTPUTS.format(files=files_str)
    
    @staticmethod
    def format_corrupted_data(path: Path) -> str:
        """Format corrupted data error message.
        
        Args:
            path: Path to the corrupted file
            
        Returns:
            Formatted error message
        """
        return ErrorMessages.CORRUPTED_DATA.format(path=path)
    
    @staticmethod
    def format_corrupted_state(path: Path, state_dir: Path) -> str:
        """Format corrupted state error message.
        
        Args:
            path: Path to the corrupted state file
            state_dir: Path to the state directory
            
        Returns:
            Formatted error message
        """
        return ErrorMessages.CORRUPTED_STATE.format(
            path=path,
            state_dir=state_dir
        )
    
    @staticmethod
    def format_permission_error(path: Path) -> str:
        """Format permission error message.
        
        Args:
            path: Path that couldn't be accessed
            
        Returns:
            Formatted error message
        """
        return ErrorMessages.PERMISSION_ERROR.format(path=path)
    
    @staticmethod
    def format_unexpected_error(error: Exception, log_path: Optional[Path] = None) -> str:
        """Format unexpected error message.
        
        Args:
            error: The exception that occurred
            log_path: Path to the log file (optional)
            
        Returns:
            Formatted error message
        """
        if log_path is None:
            log_path = Path.home() / ".mavca_ui" / "app.log"
        
        return ErrorMessages.UNEXPECTED_ERROR.format(
            error=str(error),
            log_path=log_path
        )


class WarningMessages:
    """Collection of user-friendly warning messages."""
    
    DATA_DIR_NOT_EXIST = (
        "⚠️ Data directory does not exist yet:\n{path}\n\n"
        "The directory will need to be created before running scripts.\n\n"
        "This is normal if you haven't run any modules yet for this dataset."
    )
    
    OPTIONAL_MODULE = (
        "ℹ️ This is an optional module.\n\n"
        "You can skip this module and proceed to the next required module.\n\n"
        "{description}"
    )
    
    MISSING_OPTIONAL_OUTPUTS = (
        "ℹ️ Some optional outputs are missing:\n{files}\n\n"
        "This is normal if you skipped optional scripts.\n\n"
        "The pipeline can continue without these files."
    )
    
    RERUN_WARNING = (
        "⚠️ This module has already been completed.\n\n"
        "Re-running will overwrite existing outputs.\n\n"
        "Do you want to continue?"
    )
    
    @staticmethod
    def format_data_dir_not_exist(path: Path) -> str:
        """Format data directory not exist warning.
        
        Args:
            path: Path to the non-existent directory
            
        Returns:
            Formatted warning message
        """
        return WarningMessages.DATA_DIR_NOT_EXIST.format(path=path)
    
    @staticmethod
    def format_optional_module(description: str) -> str:
        """Format optional module warning.
        
        Args:
            description: Description of what the optional module does
            
        Returns:
            Formatted warning message
        """
        return WarningMessages.OPTIONAL_MODULE.format(description=description)
    
    @staticmethod
    def format_missing_optional_outputs(files: list) -> str:
        """Format missing optional outputs warning.
        
        Args:
            files: List of missing optional output paths
            
        Returns:
            Formatted warning message
        """
        files_str = "\n".join(f"• {f}" for f in files)
        return WarningMessages.MISSING_OPTIONAL_OUTPUTS.format(files=files_str)
