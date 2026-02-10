"""
Property-based tests for ScriptExecutor class.

These tests use Hypothesis to verify universal correctness properties
across all possible inputs.
"""

import tempfile
import re
from pathlib import Path
from hypothesis import given, strategies as st, settings
from ui.models.script_executor import ScriptExecutor
from ui.models.pipeline_config import PipelineConfig


# Helper functions for validation
def _is_valid_date_format(date_str: str) -> bool:
    """Check if string matches YYYY-MM-DD format with valid values."""
    pattern = r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$'
    return bool(re.match(pattern, date_str))


def _is_valid_identifier(identifier: str) -> bool:
    """Check if string is a valid identifier."""
    if not identifier:
        return False
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, identifier))


# Feature: pipeline-ui, Property 13: Parameter Injection
@settings(max_examples=100)
@given(
    st.text(min_size=10, max_size=10).filter(lambda s: _is_valid_date_format(s)),
    st.text(min_size=1, max_size=50).filter(lambda s: _is_valid_identifier(s)),
    st.text(min_size=1, max_size=50).filter(lambda s: _is_valid_identifier(s))
)
def test_parameter_injection_property(date, mouse, run):
    """
    For any script execution, the injected DATE, MOUSE, and RUN parameters
    should match the currently configured values.
    
    Validates: Requirements 3.7
    """
    # Create a temporary virtual environment directory (doesn't need to be real for this test)
    with tempfile.TemporaryDirectory() as venv_dir:
        venv_path = Path(venv_dir)
        
        # Create a mock script with DATE, MOUSE, RUN parameters
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as script_file:
            script_content = f'''#!/usr/bin/env python3
# Mock script for testing parameter injection

DATE = "2025-01-01"
MOUSE = "original_mouse"
RUN = "original_run"

# Print parameters to verify injection
print(f"DATE={{DATE}}")
print(f"MOUSE={{MOUSE}}")
print(f"RUN={{RUN}}")
'''
            script_file.write(script_content)
            script_path = Path(script_file.name)
        
        try:
            # Create configuration with test parameters
            config = PipelineConfig()
            config.date = date
            config.mouse = mouse
            config.run = run
            
            # Create script executor
            executor = ScriptExecutor(venv_path)
            
            # Inject parameters (this creates a temporary modified script)
            modified_script_path = executor._inject_parameters(script_path, config)
            
            try:
                # Read the modified script content
                with open(modified_script_path, 'r', encoding='utf-8') as f:
                    modified_content = f.read()
                
                # Verify that DATE parameter was injected correctly
                date_pattern = r'DATE\s*=\s*"([^"]*)"'
                date_match = re.search(date_pattern, modified_content)
                assert date_match is not None, "DATE parameter not found in modified script"
                injected_date = date_match.group(1)
                assert injected_date == date, (
                    f"DATE parameter mismatch: expected '{date}', got '{injected_date}'"
                )
                
                # Verify that MOUSE parameter was injected correctly
                mouse_pattern = r'MOUSE\s*=\s*"([^"]*)"'
                mouse_match = re.search(mouse_pattern, modified_content)
                assert mouse_match is not None, "MOUSE parameter not found in modified script"
                injected_mouse = mouse_match.group(1)
                assert injected_mouse == mouse, (
                    f"MOUSE parameter mismatch: expected '{mouse}', got '{injected_mouse}'"
                )
                
                # Verify that RUN parameter was injected correctly
                run_pattern = r'RUN\s*=\s*"([^"]*)"'
                run_match = re.search(run_pattern, modified_content)
                assert run_match is not None, "RUN parameter not found in modified script"
                injected_run = run_match.group(1)
                assert injected_run == run, (
                    f"RUN parameter mismatch: expected '{run}', got '{injected_run}'"
                )
                
                # Verify that all three parameters match the configuration
                assert injected_date == config.date, "Injected DATE doesn't match config"
                assert injected_mouse == config.mouse, "Injected MOUSE doesn't match config"
                assert injected_run == config.run, "Injected RUN doesn't match config"
                
            finally:
                # Clean up modified script
                if modified_script_path.exists():
                    modified_script_path.unlink()
        
        finally:
            # Clean up original script
            if script_path.exists():
                script_path.unlink()
