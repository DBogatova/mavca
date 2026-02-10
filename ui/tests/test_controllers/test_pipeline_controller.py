"""
Property-based tests for PipelineController class.

Tests execution behavior, progress updates, and execution exclusivity.
"""

import pytest
import tempfile
import threading
import time
from pathlib import Path
from hypothesis import given, strategies as st, settings
from ui.models import PipelineConfig, ProgressState
from ui.controllers import PipelineController


# Helper strategies for generating test data
@st.composite
def valid_module_id(draw):
    """Generate a valid module ID."""
    return draw(st.sampled_from(['M1', 'M1.5', 'M2', 'M2.5', 'M3', 'M4', 'M5', 'M6']))


@st.composite
def valid_config_data(draw):
    """Generate valid configuration data."""
    # Generate valid date (YYYY-MM-DD)
    year = draw(st.integers(min_value=2020, max_value=2030))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))  # Use 28 to avoid month-specific issues
    date = f"{year:04d}-{month:02d}-{day:02d}"
    
    # Generate valid identifier (alphanumeric, underscore, hyphen)
    mouse = draw(st.text(
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        ),
        min_size=1,
        max_size=30
    ).filter(lambda s: s and s[0].isalnum()))
    
    run = draw(st.text(
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        ),
        min_size=1,
        max_size=30
    ).filter(lambda s: s and s[0].isalnum()))
    
    return date, mouse, run


@st.composite
def progress_state_with_modules(draw):
    """Generate a progress state with some completed modules."""
    dataset_key = draw(st.text(min_size=5, max_size=50))
    completed_modules = draw(st.sets(
        st.sampled_from(['M1', 'M1.5', 'M2', 'M2.5', 'M3', 'M4', 'M5', 'M6']),
        min_size=0,
        max_size=8
    ))
    m2_guided = draw(st.booleans())
    
    progress = ProgressState(dataset_key)
    for module_id in completed_modules:
        progress.mark_complete(module_id)
    progress.m2_guided = m2_guided
    
    return progress, completed_modules


# Feature: pipeline-ui, Property 11: Successful Execution Updates Progress
@given(
    config_data=valid_config_data(),
    module_id=valid_module_id(),
    initial_progress=progress_state_with_modules()
)
@settings(max_examples=100)
def test_successful_execution_updates_progress_property(tmp_path, config_data, module_id, initial_progress):
    """
    Property 11: Successful Execution Updates Progress
    
    For any successful script execution, the progress state should be updated
    to mark the corresponding module as completed.
    
    Validates: Requirements 3.4, 6.2
    """
    # Create a mock venv with python executable
    venv_path = tmp_path / "venv"
    venv_bin = venv_path / "bin"
    venv_bin.mkdir(parents=True)
    
    # Use the system python for this test
    import sys
    python_exe = venv_bin / "python"
    python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
    python_exe.chmod(0o755)
    
    # Create a mock successful script
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script_path = script_dir / "test_script.py"
    script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Script executed successfully")
''')
    
    # Create controller with temporary state directory
    state_dir = tmp_path / "state"
    controller = PipelineController()
    controller.state_manager.state_dir = state_dir
    controller.state_manager.config_file = state_dir / "config.json"
    controller.state_manager.progress_file = state_dir / "progress.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up configuration
    date, mouse, run = config_data
    controller.config.date = date
    controller.config.mouse = mouse
    controller.config.run = run
    controller.config.base_dir = tmp_path / "data"
    
    # Set up initial progress state
    initial_state, initial_completed = initial_progress
    dataset_key = f"{date}_{mouse}_{run}"
    controller.current_progress = ProgressState(dataset_key)
    for module_id_completed in initial_completed:
        controller.current_progress.mark_complete(module_id_completed)
    
    # Mock the module to use our test script
    original_module = controller.modules[module_id]
    from ui.models.module_definition import ModuleDefinition, ScriptInfo
    
    mock_module = ModuleDefinition(
        id=module_id,
        name=original_module.name,
        description=original_module.description,
        scripts=[
            ScriptInfo(
                name="test_script.py",
                path=script_path,
                description="Test script",
                optional=False
            )
        ],
        expected_outputs=original_module.expected_outputs,
        optional=original_module.optional,
        requires_interaction=original_module.requires_interaction
    )
    controller.modules[module_id] = mock_module
    
    # Record progress before execution
    was_complete_before = controller.current_progress.is_complete(module_id)
    
    # Execute the module
    output_lines = []
    try:
        controller.execute_module(module_id, lambda line: output_lines.append(line))
        execution_succeeded = True
    except Exception:
        execution_succeeded = False
    
    # Property: If execution succeeded, module should be marked complete
    if execution_succeeded:
        assert controller.current_progress.is_complete(module_id), \
            f"Module {module_id} should be marked complete after successful execution"
        
        # Verify progress was saved
        loaded_progress = controller.state_manager.load_progress(dataset_key)
        assert loaded_progress.is_complete(module_id), \
            f"Module {module_id} completion should be persisted to disk"
        
        # Verify other modules' completion status was preserved
        for other_module in initial_completed:
            if other_module != module_id:
                assert controller.current_progress.is_complete(other_module), \
                    f"Module {other_module} completion should be preserved"


# Feature: pipeline-ui, Property 12: Failed Execution Preserves Progress
@given(
    config_data=valid_config_data(),
    module_id=valid_module_id(),
    initial_progress=progress_state_with_modules()
)
@settings(max_examples=100)
def test_failed_execution_preserves_progress_property(tmp_path, config_data, module_id, initial_progress):
    """
    Property 12: Failed Execution Preserves Progress
    
    For any failed script execution, the progress state should remain
    unchanged from its pre-execution state.
    
    Validates: Requirements 3.5
    """
    # Create a mock venv with python executable
    venv_path = tmp_path / "venv"
    venv_bin = venv_path / "bin"
    venv_bin.mkdir(parents=True)
    
    # Use the system python for this test
    import sys
    python_exe = venv_bin / "python"
    python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
    python_exe.chmod(0o755)
    
    # Create a mock failing script
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script_path = script_dir / "failing_script.py"
    script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("About to fail")
raise ValueError("Intentional failure for testing")
''')
    
    # Create controller with temporary state directory
    state_dir = tmp_path / "state"
    controller = PipelineController()
    controller.state_manager.state_dir = state_dir
    controller.state_manager.config_file = state_dir / "config.json"
    controller.state_manager.progress_file = state_dir / "progress.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up configuration
    date, mouse, run = config_data
    controller.config.date = date
    controller.config.mouse = mouse
    controller.config.run = run
    controller.config.base_dir = tmp_path / "data"
    
    # Set up initial progress state
    initial_state, initial_completed = initial_progress
    dataset_key = f"{date}_{mouse}_{run}"
    controller.current_progress = ProgressState(dataset_key)
    for module_id_completed in initial_completed:
        controller.current_progress.mark_complete(module_id_completed)
    
    # Save initial progress to disk
    controller.state_manager.save_progress(controller.current_progress)
    
    # Record progress before execution
    was_complete_before = controller.current_progress.is_complete(module_id)
    completed_before = set(controller.current_progress.completed_modules)
    m2_guided_before = controller.current_progress.m2_guided
    
    # Mock the module to use our failing test script
    original_module = controller.modules[module_id]
    from ui.models.module_definition import ModuleDefinition, ScriptInfo
    
    mock_module = ModuleDefinition(
        id=module_id,
        name=original_module.name,
        description=original_module.description,
        scripts=[
            ScriptInfo(
                name="failing_script.py",
                path=script_path,
                description="Failing test script",
                optional=False
            )
        ],
        expected_outputs=original_module.expected_outputs,
        optional=original_module.optional,
        requires_interaction=original_module.requires_interaction
    )
    controller.modules[module_id] = mock_module
    
    # Execute the module (should fail)
    output_lines = []
    execution_failed = False
    try:
        controller.execute_module(module_id, lambda line: output_lines.append(line))
    except Exception:
        execution_failed = True
    
    # Property: If execution failed, progress should be unchanged
    if execution_failed:
        # Check completion status is unchanged
        is_complete_after = controller.current_progress.is_complete(module_id)
        assert is_complete_after == was_complete_before, \
            f"Module {module_id} completion status should be unchanged after failed execution"
        
        # Check all completed modules are unchanged
        completed_after = set(controller.current_progress.completed_modules)
        assert completed_after == completed_before, \
            f"Completed modules should be unchanged after failed execution"
        
        # Check m2_guided flag is unchanged
        assert controller.current_progress.m2_guided == m2_guided_before, \
            "m2_guided flag should be unchanged after failed execution"
        
        # Verify progress on disk is unchanged
        loaded_progress = controller.state_manager.load_progress(dataset_key)
        assert set(loaded_progress.completed_modules) == completed_before, \
            "Persisted progress should be unchanged after failed execution"
        assert loaded_progress.m2_guided == m2_guided_before, \
            "Persisted m2_guided flag should be unchanged after failed execution"


# Feature: pipeline-ui, Property 10: Execution Exclusivity
@given(
    config_data=valid_config_data(),
    module_id1=valid_module_id(),
    module_id2=valid_module_id()
)
@settings(max_examples=100, deadline=None)
def test_execution_exclusivity_property(tmp_path, config_data, module_id1, module_id2):
    """
    Property 10: Execution Exclusivity
    
    For any running script execution, attempting to start another script
    execution should be prevented.
    
    Validates: Requirements 3.3
    """
    # Create a mock venv with python executable
    venv_path = tmp_path / "venv"
    venv_bin = venv_path / "bin"
    venv_bin.mkdir(parents=True)
    
    # Use the system python for this test
    import sys
    python_exe = venv_bin / "python"
    python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
    python_exe.chmod(0o755)
    
    # Create a long-running script
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    long_script_path = script_dir / "long_script.py"
    long_script_path.write_text('''
import time
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Starting long script")
time.sleep(2)
print("Long script done")
''')
    
    # Create a second script
    second_script_path = script_dir / "second_script.py"
    second_script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Second script")
''')
    
    # Create controller with temporary state directory
    state_dir = tmp_path / "state"
    controller = PipelineController()
    controller.state_manager.state_dir = state_dir
    controller.state_manager.config_file = state_dir / "config.json"
    controller.state_manager.progress_file = state_dir / "progress.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up configuration
    date, mouse, run = config_data
    controller.config.date = date
    controller.config.mouse = mouse
    controller.config.run = run
    controller.config.base_dir = tmp_path / "data"
    
    # Set up progress state
    dataset_key = f"{date}_{mouse}_{run}"
    controller.current_progress = ProgressState(dataset_key)
    
    # Mock the modules to use our test scripts
    from ui.models.module_definition import ModuleDefinition, ScriptInfo
    
    original_module1 = controller.modules[module_id1]
    mock_module1 = ModuleDefinition(
        id=module_id1,
        name=original_module1.name,
        description=original_module1.description,
        scripts=[
            ScriptInfo(
                name="long_script.py",
                path=long_script_path,
                description="Long running test script",
                optional=False
            )
        ],
        expected_outputs=original_module1.expected_outputs,
        optional=original_module1.optional,
        requires_interaction=original_module1.requires_interaction
    )
    controller.modules[module_id1] = mock_module1
    
    original_module2 = controller.modules[module_id2]
    mock_module2 = ModuleDefinition(
        id=module_id2,
        name=original_module2.name,
        description=original_module2.description,
        scripts=[
            ScriptInfo(
                name="second_script.py",
                path=second_script_path,
                description="Second test script",
                optional=False
            )
        ],
        expected_outputs=original_module2.expected_outputs,
        optional=original_module2.optional,
        requires_interaction=original_module2.requires_interaction
    )
    controller.modules[module_id2] = mock_module2
    
    # Start first execution in a separate thread
    output_lines1 = []
    execution1_started = threading.Event()
    execution1_exception = None
    
    def run_first_execution():
        nonlocal execution1_exception
        try:
            execution1_started.set()
            controller.execute_module(module_id1, lambda line: output_lines1.append(line))
        except Exception as e:
            execution1_exception = e
    
    thread1 = threading.Thread(target=run_first_execution)
    thread1.start()
    
    # Wait for first execution to start
    execution1_started.wait(timeout=2)
    time.sleep(0.3)  # Give it a bit more time to actually start the subprocess
    
    # Property: Attempting to start second execution should be prevented
    output_lines2 = []
    execution2_prevented = False
    execution2_exception = None
    
    try:
        # This should raise RuntimeError because first execution is still running
        controller.execute_module(module_id2, lambda line: output_lines2.append(line))
    except RuntimeError as e:
        if "already running" in str(e).lower():
            execution2_prevented = True
            execution2_exception = e
    except Exception as e:
        execution2_exception = e
    
    # Wait for first execution to complete
    thread1.join(timeout=5)
    
    # Verify the property
    assert execution2_prevented, \
        f"Second execution should be prevented when first execution is running. " \
        f"Exception: {execution2_exception}"
    
    # Verify first execution completed (if no exception)
    if execution1_exception is None:
        assert controller.current_progress.is_complete(module_id1), \
            f"First module {module_id1} should be marked complete"
    
    # Verify second execution did not complete
    assert not controller.current_progress.is_complete(module_id2), \
        f"Second module {module_id2} should not be marked complete since it was prevented"


# Additional unit tests for edge cases
class TestPipelineControllerExecutionBehavior:
    """Unit tests for specific execution behavior scenarios."""
    
    def test_successful_execution_marks_module_complete(self, tmp_path):
        """Test that successful execution marks module as complete."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        # Create a successful script
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        script_path = script_dir / "success.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Success")
''')
        
        # Create controller
        state_dir = tmp_path / "state"
        controller = PipelineController()
        controller.state_manager.state_dir = state_dir
        controller.state_manager.config_file = state_dir / "config.json"
        controller.state_manager.progress_file = state_dir / "progress.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        controller.config.date = "2025-01-15"
        controller.config.mouse = "test_mouse"
        controller.config.run = "run1"
        controller.config.base_dir = tmp_path / "data"
        
        dataset_key = "2025-01-15_test_mouse_run1"
        controller.current_progress = ProgressState(dataset_key)
        
        # Mock M1 module
        from ui.models.module_definition import ModuleDefinition, ScriptInfo
        mock_module = ModuleDefinition(
            id="M1",
            name="Test Module",
            description="Test",
            scripts=[
                ScriptInfo(
                    name="success.py",
                    path=script_path,
                    description="Success script",
                    optional=False
                )
            ],
            expected_outputs=[],
            optional=False,
            requires_interaction=False
        )
        controller.modules["M1"] = mock_module
        
        # Execute
        assert not controller.current_progress.is_complete("M1")
        controller.execute_module("M1", lambda x: None)
        assert controller.current_progress.is_complete("M1")
    
    def test_failed_execution_does_not_mark_complete(self, tmp_path):
        """Test that failed execution does not mark module as complete."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        # Create a failing script
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        script_path = script_dir / "fail.py"
        script_path.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
raise ValueError("Fail")
''')
        
        # Create controller
        state_dir = tmp_path / "state"
        controller = PipelineController()
        controller.state_manager.state_dir = state_dir
        controller.state_manager.config_file = state_dir / "config.json"
        controller.state_manager.progress_file = state_dir / "progress.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        controller.config.date = "2025-01-15"
        controller.config.mouse = "test_mouse"
        controller.config.run = "run1"
        controller.config.base_dir = tmp_path / "data"
        
        dataset_key = "2025-01-15_test_mouse_run1"
        controller.current_progress = ProgressState(dataset_key)
        
        # Mock M1 module
        from ui.models.module_definition import ModuleDefinition, ScriptInfo
        mock_module = ModuleDefinition(
            id="M1",
            name="Test Module",
            description="Test",
            scripts=[
                ScriptInfo(
                    name="fail.py",
                    path=script_path,
                    description="Failing script",
                    optional=False
                )
            ],
            expected_outputs=[],
            optional=False,
            requires_interaction=False
        )
        controller.modules["M1"] = mock_module
        
        # Execute (should fail)
        assert not controller.current_progress.is_complete("M1")
        with pytest.raises(Exception):
            controller.execute_module("M1", lambda x: None)
        assert not controller.current_progress.is_complete("M1")
    
    def test_cannot_start_execution_while_running(self, tmp_path):
        """Test that starting execution while running raises RuntimeError."""
        # Create a mock venv with python executable
        venv_path = tmp_path / "venv"
        venv_bin = venv_path / "bin"
        venv_bin.mkdir(parents=True)
        
        import sys
        python_exe = venv_bin / "python"
        python_exe.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        python_exe.chmod(0o755)
        
        # Create a long-running script
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        long_script = script_dir / "long.py"
        long_script.write_text('''
import time
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
time.sleep(1)
''')
        
        short_script = script_dir / "short.py"
        short_script.write_text('''
DATE = "2025-01-01"
MOUSE = "mouse"
RUN = "run"
print("Short")
''')
        
        # Create controller
        state_dir = tmp_path / "state"
        controller = PipelineController()
        controller.state_manager.state_dir = state_dir
        controller.state_manager.config_file = state_dir / "config.json"
        controller.state_manager.progress_file = state_dir / "progress.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        controller.config.date = "2025-01-15"
        controller.config.mouse = "test_mouse"
        controller.config.run = "run1"
        controller.config.base_dir = tmp_path / "data"
        
        dataset_key = "2025-01-15_test_mouse_run1"
        controller.current_progress = ProgressState(dataset_key)
        
        # Mock modules
        from ui.models.module_definition import ModuleDefinition, ScriptInfo
        controller.modules["M1"] = ModuleDefinition(
            id="M1",
            name="Long Module",
            description="Test",
            scripts=[ScriptInfo("long.py", long_script, "Long", False)],
            expected_outputs=[],
            optional=False,
            requires_interaction=False
        )
        controller.modules["M2"] = ModuleDefinition(
            id="M2",
            name="Short Module",
            description="Test",
            scripts=[ScriptInfo("short.py", short_script, "Short", False)],
            expected_outputs=[],
            optional=False,
            requires_interaction=False
        )
        
        # Start first execution in thread
        def run_long():
            controller.execute_module("M1", lambda x: None)
        
        thread = threading.Thread(target=run_long)
        thread.start()
        time.sleep(0.2)  # Let it start
        
        # Try to start second execution
        with pytest.raises(RuntimeError, match="already running"):
            controller.execute_module("M2", lambda x: None)
        
        thread.join(timeout=3)
