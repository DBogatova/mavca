"""
Script execution management for the MAVCA Pipeline UI.

This module provides the ScriptExecutor class for executing Python scripts
with real-time output capture and parameter injection.
"""

import subprocess
import tempfile
import re
from pathlib import Path
from typing import Optional, Callable
from .pipeline_config import PipelineConfig


class ScriptExecutor:
    """Executes Python scripts and captures output.
    
    Handles subprocess-based script execution with real-time output capture,
    parameter injection, and process termination capabilities.
    
    Attributes:
        venv_path: Path to the virtual environment directory
        process: Currently running subprocess (if any)
        output_callback: Callback function for real-time output
    """
    
    def __init__(self, venv_path: Path):
        """Initialize the script executor.
        
        Args:
            venv_path: Path to the virtual environment directory
        """
        self.venv_path: Path = venv_path
        self.process: Optional[subprocess.Popen] = None
        self.output_callback: Optional[Callable[[str], None]] = None
    
    def execute_script(
        self,
        script_path: Path,
        config: PipelineConfig,
        output_callback: Callable[[str], None]
    ) -> int:
        """Execute a script with given configuration.
        
        Executes a Python script using the virtual environment interpreter,
        injecting the DATE, MOUSE, and RUN parameters from the configuration.
        Captures stdout and stderr in real-time and calls the output_callback
        for each line of output.
        
        Args:
            script_path: Path to the Python script to execute
            config: Pipeline configuration with DATE, MOUSE, RUN parameters
            output_callback: Callback function to receive output lines
            
        Returns:
            Exit code from the script execution (0 for success)
            
        Raises:
            FileNotFoundError: If script_path or Python interpreter not found
            RuntimeError: If a script is already running
        """
        if self.is_running():
            raise RuntimeError("A script is already running")
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Find Python interpreter in virtual environment
        python_executable = self._find_python_interpreter()
        if not python_executable.exists():
            raise FileNotFoundError(
                f"Python interpreter not found in virtual environment: {python_executable}"
            )
        
        # Inject parameters into script
        modified_script_path = self._inject_parameters(script_path, config)
        
        self.output_callback = output_callback
        
        try:
            output_callback(f"[DEBUG] Python: {python_executable}")
            output_callback(f"[DEBUG] Script: {modified_script_path}")
            
            # Start the subprocess
            self.process = subprocess.Popen(
                [str(python_executable), "-u", str(modified_script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=str(script_path.parent)
            )
            
            # Read output in real-time
            if self.process.stdout:
                for line in self.process.stdout:
                    if self.output_callback:
                        self.output_callback(line.rstrip('\n'))
            
            # Wait for process to complete
            exit_code = self.process.wait()
            
            return exit_code
            
        finally:
            # Clean up temporary modified script
            if modified_script_path.exists():
                modified_script_path.unlink()
            self.process = None
            self.output_callback = None
    
    def terminate(self) -> None:
        """Terminate the running process.
        
        Sends a termination signal to the currently running process.
        Does nothing if no process is running.
        """
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                # Wait up to 5 seconds for graceful termination
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                self.process.kill()
                self.process.wait()
    
    def is_running(self) -> bool:
        """Check if a process is currently running.
        
        Returns:
            True if a process is running, False otherwise
        """
        return self.process is not None and self.process.poll() is None
    
    def _inject_parameters(
        self,
        script_path: Path,
        config: PipelineConfig
    ) -> Path:
        """Create a temporary modified script with injected parameters.
        
        This allows parameter override without modifying original scripts.
        Searches for DATE, MOUSE, and RUN variable assignments in the script
        and replaces their values with those from the configuration.
        
        Args:
            script_path: Path to the original script
            config: Pipeline configuration with parameters to inject
            
        Returns:
            Path to the temporary modified script
        """
        # Read the original script
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Replace DATE, MOUSE, RUN parameters
        # Pattern matches: DATE = "value" or DATE = 'value'
        # Handles various quote styles and whitespace
        
        # Replace DATE
        date_pattern = r'(DATE\s*=\s*)["\']([^"\']*)["\']'
        script_content = re.sub(
            date_pattern,
            rf'\1"{config.date}"',
            script_content
        )
        
        # Replace MOUSE
        mouse_pattern = r'(MOUSE\s*=\s*)["\']([^"\']*)["\']'
        script_content = re.sub(
            mouse_pattern,
            rf'\1"{config.mouse}"',
            script_content
        )
        
        # Replace RUN
        run_pattern = r'(RUN\s*=\s*)["\']([^"\']*)["\']'
        script_content = re.sub(
            run_pattern,
            rf'\1"{config.run}"',
            script_content
        )
        
        # Create temporary file with modified content
        # Use the same directory as the original script to maintain relative imports
        temp_dir = script_path.parent
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            prefix=f'_temp_{script_path.stem}_',
            dir=temp_dir,
            delete=False,
            encoding='utf-8'
        )
        
        temp_file.write(script_content)
        temp_file.close()
        
        return Path(temp_file.name)
    
    def _find_python_interpreter(self) -> Path:
        """Find the Python interpreter in the virtual environment.
        
        Searches for the Python executable in the virtual environment's
        bin directory (Unix/macOS) or Scripts directory (Windows).
        
        Returns:
            Path to the Python interpreter
        """
        # Try Unix/macOS path first
        python_path = self.venv_path / "bin" / "python"
        if python_path.exists():
            return python_path
        
        # Try alternative Unix/macOS paths
        python3_path = self.venv_path / "bin" / "python3"
        if python3_path.exists():
            return python3_path
        
        # Try Windows path
        windows_path = self.venv_path / "Scripts" / "python.exe"
        if windows_path.exists():
            return windows_path
        
        # Return the first path as default (will fail with FileNotFoundError later)
        return python_path
