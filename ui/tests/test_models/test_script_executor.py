"""
Unit tests for ScriptExecutor class.

Tests output capture, termination functionality, and virtual environment detection.
"""

import pytest
import tempfile
import time
from pathlib import Path
from ui.models import ScriptExecutor, PipelineConfig


class TestScriptExecutor:
    """Test suite for ScriptExecutor class."""
    
    def test_init_sets_venv_path(self, tmp_path):
        """Test that ScriptExecutor initializes with venv path."""
        venv_path = tmp_path / "venv"
        executor = ScriptExecutor(venv_path)
        
        assert executor.venv_path == venv_path
        assert executor.process is None
        assert executor.output_callback is None
    
    def test_is_running_returns_false_initially(self, tmp_path):
        """Test that is_running returns False when no process is running."""
        venv_path = tmp_path / "venv"
        executor = ScriptExecutor(venv_path)
        
        assert executor.is_running() is False
    
    def test_execute_script_raises_when_script_not_found(self, tmp_path):
        """Test that execute_script raises FileNotFoundError for missing script."""
        venv_path = tmp_path / "venv"
        executor = ScriptExecutor(venv_path)
        
        config = PipelineConfig()
        config.date = "2025-01-15"
        config.mouse = "test_mouse"
        config.run = "run1"
        
        non_existent_script = tmp_path / "non_existent.py"
        
        with pytest.raises(FileNotFoundError, match="Script not found"):
            executor.execute_script(non_existent_script, config, lambda x: None)
    
    def test_execute_script_raises_when_already_running(self, tmp_path):
        """Test that execute_script raises RuntimeError when already running."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Create a simple mock python script that acts as interpreter
        python_exe = venv_bin / "python"
        python_exe.write_text("#!/bin/bash\nsleep 0.1\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        # Mock is_running to return True
        executor.process = object()  # Non-None value
        
        config = PipelineConfig()
        script_path = tmp_path / "test.py"
        script_path.write_text("print('test')")
        
        with pytest.raises(RuntimeError, match="A script is already running"):
            executor.execute_script(script_path, config, lambda x: None)
    
    def test_find_python_interpreter_unix_bin_python(self, tmp_path):
        """Test finding Python interpreter in Unix/macOS bin/python path."""
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        python_exe = venv_bin / "python"
        python_exe.touch()
        
        executor = ScriptExecutor(venv_path)
        found_python = executor._find_python_interpreter()
        
        assert found_python == python_exe
    
    def test_find_python_interpreter_unix_bin_python3(self, tmp_path):
        """Test finding Python interpreter in Unix/macOS bin/python3 path."""
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Don't create bin/python, only bin/python3
        python3_exe = venv_bin / "python3"
        python3_exe.touch()
        
        executor = ScriptExecutor(venv_path)
        found_python = executor._find_python_interpreter()
        
        assert found_python == python3_exe
    
    def test_find_python_interpreter_windows_scripts(self, tmp_path):
        """Test finding Python interpreter in Windows Scripts/python.exe path."""
        venv_path = tmp_path / "venv"
        venv_scripts = venv_path / "Scripts"
        venv_scripts.mkdir(parents=True)
        
        python_exe = venv_scripts / "python.exe"
        python_exe.touch()
        
        executor = ScriptExecutor(venv_path)
        found_python = executor._find_python_interpreter()
        
        assert found_python == python_exe
    
    def test_find_python_interpreter_returns_default_when_not_found(self, tmp_path):
        """Test that _find_python_interpreter returns default path when not found."""
        venv_path = tmp_path / "venv"
        venv_path.mkdir()
        
        executor = ScriptExecutor(venv_path)
        found_python = executor._find_python_interpreter()
        
        # Should return the first default path (bin/python)
        assert found_python == venv_path / "bin" / "python"
    
    def test_inject_parameters_replaces_date(self, tmp_path):
        """Test that _inject_parameters replaces DATE parameter."""
        executor = ScriptExecutor(tmp_path / "venv")
        
        script_path = tmp_path / "test_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "original_mouse"
RUN = "original_run"
print(f"Processing {DATE} {MOUSE} {RUN}")
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "original_mouse"
        config.run = "original_run"
        
        modified_path = executor._inject_parameters(script_path, config)
        
        try:
            modified_content = modified_path.read_text()
            
            assert 'DATE = "2025-12-25"' in modified_content
            assert 'DATE = "2025-01-01"' not in modified_content
        finally:
            if modified_path.exists():
                modified_path.unlink()
    
    def test_inject_parameters_replaces_mouse(self, tmp_path):
        """Test that _inject_parameters replaces MOUSE parameter."""
        executor = ScriptExecutor(tmp_path / "venv")
        
        script_path = tmp_path / "test_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "original_mouse"
RUN = "original_run"
''')
        
        config = PipelineConfig()
        config.date = "2025-01-01"
        config.mouse = "new_mouse_123"
        config.run = "original_run"
        
        modified_path = executor._inject_parameters(script_path, config)
        
        try:
            modified_content = modified_path.read_text()
            
            assert 'MOUSE = "new_mouse_123"' in modified_content
            assert 'MOUSE = "original_mouse"' not in modified_content
        finally:
            if modified_path.exists():
                modified_path.unlink()
    
    def test_inject_parameters_replaces_run(self, tmp_path):
        """Test that _inject_parameters replaces RUN parameter."""
        executor = ScriptExecutor(tmp_path / "venv")
        
        script_path = tmp_path / "test_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "original_mouse"
RUN = "original_run"
''')
        
        config = PipelineConfig()
        config.date = "2025-01-01"
        config.mouse = "original_mouse"
        config.run = "run_999"
        
        modified_path = executor._inject_parameters(script_path, config)
        
        try:
            modified_content = modified_path.read_text()
            
            assert 'RUN = "run_999"' in modified_content
            assert 'RUN = "original_run"' not in modified_content
        finally:
            if modified_path.exists():
                modified_path.unlink()
    
    def test_inject_parameters_replaces_all_parameters(self, tmp_path):
        """Test that _inject_parameters replaces all parameters simultaneously."""
        executor = ScriptExecutor(tmp_path / "venv")
        
        script_path = tmp_path / "test_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "original_mouse"
RUN = "original_run"
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "new_mouse"
        config.run = "new_run"
        
        modified_path = executor._inject_parameters(script_path, config)
        
        try:
            modified_content = modified_path.read_text()
            
            assert 'DATE = "2025-12-25"' in modified_content
            assert 'MOUSE = "new_mouse"' in modified_content
            assert 'RUN = "new_run"' in modified_content
            
            # Verify old values are gone
            assert 'DATE = "2025-01-01"' not in modified_content
            assert 'MOUSE = "original_mouse"' not in modified_content
            assert 'RUN = "original_run"' not in modified_content
        finally:
            if modified_path.exists():
                modified_path.unlink()
    
    def test_inject_parameters_handles_single_quotes(self, tmp_path):
        """Test that _inject_parameters handles single-quoted strings."""
        executor = ScriptExecutor(tmp_path / "venv")
        
        script_path = tmp_path / "test_script.py"
        script_path.write_text("""
DATE = '2025-01-01'
MOUSE = 'original_mouse'
RUN = 'original_run'
""")
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "new_mouse"
        config.run = "new_run"
        
        modified_path = executor._inject_parameters(script_path, config)
        
        try:
            modified_content = modified_path.read_text()
            
            # Should replace with double quotes
            assert 'DATE = "2025-12-25"' in modified_content
            assert 'MOUSE = "new_mouse"' in modified_content
            assert 'RUN = "new_run"' in modified_content
        finally:
            if modified_path.exists():
                modified_path.unlink()
    
    def test_inject_parameters_handles_whitespace_variations(self, tmp_path):
        """Test that _inject_parameters handles various whitespace patterns."""
        executor = ScriptExecutor(tmp_path / "venv")
        
        script_path = tmp_path / "test_script.py"
        script_path.write_text('''
DATE="2025-01-01"
MOUSE  =  "original_mouse"
RUN   =   "original_run"
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "new_mouse"
        config.run = "new_run"
        
        modified_path = executor._inject_parameters(script_path, config)
        
        try:
            modified_content = modified_path.read_text()
            
            # Should handle all whitespace variations
            assert 'DATE="2025-12-25"' in modified_content or 'DATE = "2025-12-25"' in modified_content
            assert '"new_mouse"' in modified_content
            assert '"new_run"' in modified_content
        finally:
            if modified_path.exists():
                modified_path.unlink()
    
    def test_inject_parameters_preserves_other_content(self, tmp_path):
        """Test that _inject_parameters preserves non-parameter content."""
        executor = ScriptExecutor(tmp_path / "venv")
        
        script_path = tmp_path / "test_script.py"
        script_path.write_text('''
# This is a comment
import sys
import os

DATE = "2025-01-01"
MOUSE = "original_mouse"
RUN = "original_run"

def process_data():
    """Process the data."""
    print(f"Processing {DATE} {MOUSE} {RUN}")
    return True

if __name__ == "__main__":
    process_data()
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "new_mouse"
        config.run = "new_run"
        
        modified_path = executor._inject_parameters(script_path, config)
        
        try:
            modified_content = modified_path.read_text()
            
            # Verify parameters are replaced
            assert 'DATE = "2025-12-25"' in modified_content
            assert 'MOUSE = "new_mouse"' in modified_content
            assert 'RUN = "new_run"' in modified_content
            
            # Verify other content is preserved
            assert '# This is a comment' in modified_content
            assert 'import sys' in modified_content
            assert 'import os' in modified_content
            assert 'def process_data():' in modified_content
            assert '"""Process the data."""' in modified_content
            assert 'if __name__ == "__main__":' in modified_content
        finally:
            if modified_path.exists():
                modified_path.unlink()
    
    def test_inject_parameters_creates_temp_file_in_same_directory(self, tmp_path):
        """Test that _inject_parameters creates temp file in same directory as original."""
        executor = ScriptExecutor(tmp_path / "venv")
        
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        script_path = script_dir / "test_script.py"
        script_path.write_text('DATE = "2025-01-01"\n')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "mouse"
        config.run = "run"
        
        modified_path = executor._inject_parameters(script_path, config)
        
        try:
            # Verify temp file is in same directory
            assert modified_path.parent == script_dir
            assert modified_path.name.startswith('_temp_test_script_')
            assert modified_path.suffix == '.py'
        finally:
            if modified_path.exists():
                modified_path.unlink()
    
    def test_inject_parameters_cleans_up_temp_file(self, tmp_path):
        """Test that execute_script cleans up temporary modified script."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Use the system python for this test
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        script_path = tmp_path / "test_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Hello")
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "new_mouse"
        config.run = "new_run"
        
        # Track temp files before execution
        temp_files_before = list(tmp_path.glob('_temp_*.py'))
        
        # Execute script
        output_lines = []
        exit_code = executor.execute_script(
            script_path,
            config,
            lambda line: output_lines.append(line)
        )
        
        # Track temp files after execution
        temp_files_after = list(tmp_path.glob('_temp_*.py'))
        
        # Verify temp file was cleaned up
        assert len(temp_files_after) == len(temp_files_before), \
            "Temporary file should be cleaned up after execution"
    
    def test_terminate_stops_running_process(self, tmp_path):
        """Test that terminate stops a running process."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Use the system python for this test
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        # Create a long-running script
        script_path = tmp_path / "long_script.py"
        script_path.write_text('''
import time
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Starting")
time.sleep(10)
print("Done")
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "mouse"
        config.run = "run"
        
        # Start execution in a separate thread
        import threading
        output_lines = []
        
        def run_script():
            executor.execute_script(
                script_path,
                config,
                lambda line: output_lines.append(line)
            )
        
        thread = threading.Thread(target=run_script)
        thread.start()
        
        # Wait a bit for script to start
        time.sleep(0.5)
        
        # Verify process is running
        assert executor.is_running()
        
        # Terminate the process
        executor.terminate()
        
        # Wait for thread to finish
        thread.join(timeout=2)
        
        # Verify process is no longer running
        assert not executor.is_running()
    
    def test_terminate_does_nothing_when_no_process(self, tmp_path):
        """Test that terminate does nothing when no process is running."""
        venv_path = tmp_path / "venv"
        executor = ScriptExecutor(venv_path)
        
        # Should not raise any exception
        executor.terminate()
        
        assert not executor.is_running()
    
    def test_output_capture_with_mock_script(self, tmp_path):
        """Test output capture with a mock script that prints multiple lines."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Use the system python for this test
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        # Create a script that prints multiple lines
        script_path = tmp_path / "output_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Line 1")
print("Line 2")
print("Line 3")
import sys
print("Error line", file=sys.stderr)
print("Line 4")
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "mouse"
        config.run = "run"
        
        output_lines = []
        exit_code = executor.execute_script(
            script_path,
            config,
            lambda line: output_lines.append(line)
        )
        
        # Verify all output was captured
        assert exit_code == 0
        assert "Line 1" in output_lines
        assert "Line 2" in output_lines
        assert "Line 3" in output_lines
        assert "Line 4" in output_lines
        # stderr should also be captured (stdout and stderr are merged)
        assert "Error line" in output_lines
    
    def test_output_capture_real_time(self, tmp_path):
        """Test that output is captured in real-time, not buffered."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Use the system python for this test
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        # Create a script that prints with delays
        script_path = tmp_path / "timed_script.py"
        script_path.write_text('''
import time
import sys
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Start", flush=True)
sys.stdout.flush()
time.sleep(0.1)
print("Middle", flush=True)
sys.stdout.flush()
time.sleep(0.1)
print("End", flush=True)
sys.stdout.flush()
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "mouse"
        config.run = "run"
        
        output_lines = []
        timestamps = []
        
        def capture_with_timestamp(line):
            output_lines.append(line)
            timestamps.append(time.time())
        
        start_time = time.time()
        exit_code = executor.execute_script(
            script_path,
            config,
            capture_with_timestamp
        )
        
        # Verify output was captured
        assert exit_code == 0
        assert "Start" in output_lines
        assert "Middle" in output_lines
        assert "End" in output_lines
        
        # Verify output came in real-time (not all at once at the end)
        # The timestamps should be spread out, not all at the same time
        if len(timestamps) >= 3:
            # Check that there's some time difference between captures
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            # At least one time difference should be > 0.05 seconds
            assert any(diff > 0.05 for diff in time_diffs), \
                "Output should be captured in real-time, not buffered"
    
    def test_execute_script_returns_exit_code_zero_on_success(self, tmp_path):
        """Test that execute_script returns exit code 0 on successful execution."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Use the system python for this test
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        script_path = tmp_path / "success_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Success")
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "mouse"
        config.run = "run"
        
        exit_code = executor.execute_script(script_path, config, lambda x: None)
        
        assert exit_code == 0
    
    def test_execute_script_returns_nonzero_exit_code_on_failure(self, tmp_path):
        """Test that execute_script returns non-zero exit code on script failure."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Use the system python for this test
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        script_path = tmp_path / "failure_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("About to fail")
raise ValueError("Intentional error")
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "mouse"
        config.run = "run"
        
        output_lines = []
        exit_code = executor.execute_script(
            script_path,
            config,
            lambda line: output_lines.append(line)
        )
        
        assert exit_code != 0
        # Verify error output was captured
        assert any("ValueError" in line or "Intentional error" in line for line in output_lines)
    
    def test_execute_script_clears_process_after_completion(self, tmp_path):
        """Test that execute_script clears process reference after completion."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Use the system python for this test
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        script_path = tmp_path / "simple_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Done")
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "mouse"
        config.run = "run"
        
        executor.execute_script(script_path, config, lambda x: None)
        
        # Process should be cleared after execution
        assert executor.process is None
        assert not executor.is_running()
    
    def test_execute_script_clears_output_callback_after_completion(self, tmp_path):
        """Test that execute_script clears output callback after completion."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        # Use the system python for this test
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        executor = ScriptExecutor(venv_path)
        
        script_path = tmp_path / "simple_script.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Done")
''')
        
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "mouse"
        config.run = "run"
        
        executor.execute_script(script_path, config, lambda x: None)
        
        # Output callback should be cleared after execution
        assert executor.output_callback is None
