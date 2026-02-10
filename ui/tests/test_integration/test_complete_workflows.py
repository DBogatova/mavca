"""
Integration tests for complete workflows.

Tests end-to-end workflows including:
- Complete M1-M6 pipeline execution
- M2/M2.5 iteration workflow
- Multi-dataset switching
- State persistence across restarts
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from ui.models.pipeline_config import PipelineConfig
from ui.models.progress_state import ProgressState
from ui.models.state_manager import StateManager
from ui.controllers.pipeline_controller import PipelineController


@pytest.fixture
def temp_state_dir():
    """Create a temporary state directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_complete_pipeline_workflow_simulation(temp_state_dir, temp_data_dir):
    """
    Test simulating a complete M1-M6 pipeline workflow.
    
    This test simulates the workflow without actually executing scripts,
    focusing on state management and progress tracking.
    """
    # Create state manager
    state_manager = StateManager(temp_state_dir)
    
    # Create configuration
    config = PipelineConfig()
    config.date = "2025-01-01"
    config.mouse = "test_mouse"
    config.run = "run1"
    config.base_dir = temp_data_dir
    
    # Save configuration
    state_manager.save_config(config)
    
    # Create progress state
    dataset_key = f"{config.date}_{config.mouse}_{config.run}"
    progress = ProgressState(dataset_key)
    
    # Simulate module completions
    modules = ["M1", "M1.5", "M2", "M2.5", "M2", "M3", "M4", "M5", "M6"]
    
    for module_id in modules:
        progress.mark_complete(module_id)
        
        # Special handling for M2 after M2.5
        if module_id == "M2" and progress.is_complete("M2.5"):
            progress.m2_guided = True
        
        # Save progress after each module
        state_manager.save_progress(progress)
    
    # Verify all modules completed
    for module_id in modules:
        assert progress.is_complete(module_id), \
            f"Module {module_id} should be completed"
    
    # Verify M2 was run with guides
    assert progress.m2_guided is True, \
        "M2 should be marked as guided after M2.5"
    
    # Load progress and verify persistence
    loaded_progress = state_manager.load_progress(dataset_key)
    assert loaded_progress.completed_modules == progress.completed_modules
    assert loaded_progress.m2_guided == progress.m2_guided


def test_m2_m2_5_iteration_workflow(temp_state_dir, temp_data_dir):
    """
    Test M2/M2.5 iteration workflow.
    
    Simulates:
    1. Run M2 (non-guided)
    2. Run M2.5 (add guides)
    3. Re-run M2 (guided)
    """
    state_manager = StateManager(temp_state_dir)
    
    config = PipelineConfig()
    config.date = "2025-01-01"
    config.mouse = "test_mouse"
    config.run = "run1"
    config.base_dir = temp_data_dir
    
    dataset_key = f"{config.date}_{config.mouse}_{config.run}"
    progress = ProgressState(dataset_key)
    
    # Step 1: Run M2 (non-guided)
    progress.mark_complete("M2")
    progress.m2_guided = False
    state_manager.save_progress(progress)
    
    assert progress.is_complete("M2")
    assert progress.m2_guided is False
    
    # Step 2: Run M2.5 (add guides)
    progress.mark_complete("M2.5")
    state_manager.save_progress(progress)
    
    assert progress.is_complete("M2.5")
    
    # Step 3: Re-run M2 (guided)
    progress.mark_complete("M2")  # Mark complete again
    progress.m2_guided = True  # Now guided
    state_manager.save_progress(progress)
    
    assert progress.is_complete("M2")
    assert progress.m2_guided is True
    
    # Verify persistence
    loaded_progress = state_manager.load_progress(dataset_key)
    assert loaded_progress.is_complete("M2")
    assert loaded_progress.is_complete("M2.5")
    assert loaded_progress.m2_guided is True


def test_multi_dataset_switching(temp_state_dir, temp_data_dir):
    """
    Test switching between multiple datasets.
    
    Verifies that progress is correctly loaded for each dataset.
    """
    state_manager = StateManager(temp_state_dir)
    
    # Create two datasets
    datasets = [
        ("2025-01-01", "mouse1", "run1"),
        ("2025-01-02", "mouse2", "run2")
    ]
    
    progress_states = {}
    
    # Create progress for each dataset
    for date, mouse, run in datasets:
        dataset_key = f"{date}_{mouse}_{run}"
        progress = ProgressState(dataset_key)
        
        # Complete different modules for each dataset
        if dataset_key.endswith("run1"):
            progress.mark_complete("M1")
            progress.mark_complete("M2")
        else:
            progress.mark_complete("M1")
            progress.mark_complete("M1.5")
            progress.mark_complete("M2")
            progress.mark_complete("M3")
        
        state_manager.save_progress(progress)
        progress_states[dataset_key] = progress
    
    # Switch between datasets and verify progress
    for date, mouse, run in datasets:
        dataset_key = f"{date}_{mouse}_{run}"
        loaded_progress = state_manager.load_progress(dataset_key)
        
        # Verify loaded progress matches saved progress
        original_progress = progress_states[dataset_key]
        assert loaded_progress.completed_modules == original_progress.completed_modules
        assert loaded_progress.dataset_key == original_progress.dataset_key


def test_state_persistence_across_restarts(temp_state_dir, temp_data_dir):
    """
    Test state persistence across application restarts.
    
    Simulates:
    1. Configure and run some modules
    2. Save state
    3. Create new controller (simulating restart)
    4. Verify state is restored
    """
    # First "session"
    state_manager1 = StateManager(temp_state_dir)
    
    config1 = PipelineConfig()
    config1.date = "2025-01-01"
    config1.mouse = "test_mouse"
    config1.run = "run1"
    config1.base_dir = temp_data_dir
    
    state_manager1.save_config(config1)
    
    dataset_key = f"{config1.date}_{config1.mouse}_{config1.run}"
    progress1 = ProgressState(dataset_key)
    progress1.mark_complete("M1")
    progress1.mark_complete("M2")
    state_manager1.save_progress(progress1)
    
    # Second "session" (simulating restart)
    state_manager2 = StateManager(temp_state_dir)
    
    # Load configuration
    config2 = state_manager2.load_config()
    assert config2 is not None
    assert config2.date == config1.date
    assert config2.mouse == config1.mouse
    assert config2.run == config1.run
    
    # Load progress
    progress2 = state_manager2.load_progress(dataset_key)
    assert progress2.is_complete("M1")
    assert progress2.is_complete("M2")
    assert not progress2.is_complete("M3")


def test_progress_reset_workflow(temp_state_dir, temp_data_dir):
    """
    Test resetting progress for a dataset.
    
    Verifies that reset clears all progress but preserves configuration.
    """
    state_manager = StateManager(temp_state_dir)
    
    config = PipelineConfig()
    config.date = "2025-01-01"
    config.mouse = "test_mouse"
    config.run = "run1"
    config.base_dir = temp_data_dir
    
    state_manager.save_config(config)
    
    dataset_key = f"{config.date}_{config.mouse}_{config.run}"
    progress = ProgressState(dataset_key)
    
    # Complete some modules
    progress.mark_complete("M1")
    progress.mark_complete("M2")
    progress.mark_complete("M3")
    progress.m2_guided = True
    state_manager.save_progress(progress)
    
    # Reset progress
    progress.reset()
    state_manager.save_progress(progress)
    
    # Verify reset
    assert len(progress.completed_modules) == 0
    assert progress.m2_guided is False
    
    # Verify configuration still exists
    loaded_config = state_manager.load_config()
    assert loaded_config is not None
    assert loaded_config.date == config.date


def test_output_verification_workflow(temp_data_dir):
    """
    Test output verification workflow.
    
    Simulates checking for expected outputs after module completion.
    """
    # Create a mock data directory structure
    config = PipelineConfig()
    config.date = "2025-01-01"
    config.mouse = "test_mouse"
    config.run = "run1"
    config.base_dir = temp_data_dir
    
    data_path = config.get_data_path()
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create some expected outputs for M1
    preprocessed_dir = data_path / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)
    
    (preprocessed_dir / "raw_clean.tif").touch()
    (preprocessed_dir / "event_crops").mkdir(exist_ok=True)
    
    # Verify outputs exist
    assert (preprocessed_dir / "raw_clean.tif").exists()
    assert (preprocessed_dir / "event_crops").exists()
    
    # Simulate missing output
    assert not (preprocessed_dir / "missing_file.npy").exists()


def test_configuration_update_workflow(temp_state_dir, temp_data_dir):
    """
    Test updating configuration and loading corresponding progress.
    
    Simulates changing DATE/MOUSE/RUN and verifying correct progress loads.
    """
    state_manager = StateManager(temp_state_dir)
    
    # Create progress for two datasets
    datasets = [
        ("2025-01-01", "mouse1", "run1", ["M1", "M2"]),
        ("2025-01-02", "mouse2", "run2", ["M1", "M1.5", "M2", "M3"])
    ]
    
    for date, mouse, run, completed_modules in datasets:
        dataset_key = f"{date}_{mouse}_{run}"
        progress = ProgressState(dataset_key)
        for module_id in completed_modules:
            progress.mark_complete(module_id)
        state_manager.save_progress(progress)
    
    # Simulate configuration updates
    for date, mouse, run, completed_modules in datasets:
        config = PipelineConfig()
        config.date = date
        config.mouse = mouse
        config.run = run
        config.base_dir = temp_data_dir
        
        dataset_key = f"{date}_{mouse}_{run}"
        progress = state_manager.load_progress(dataset_key)
        
        # Verify correct progress loaded
        for module_id in completed_modules:
            assert progress.is_complete(module_id), \
                f"Module {module_id} should be complete for {dataset_key}"


def test_error_recovery_workflow(temp_state_dir, temp_data_dir):
    """
    Test error recovery workflow.
    
    Simulates a module execution failure and verifies progress is preserved.
    """
    state_manager = StateManager(temp_state_dir)
    
    config = PipelineConfig()
    config.date = "2025-01-01"
    config.mouse = "test_mouse"
    config.run = "run1"
    config.base_dir = temp_data_dir
    
    dataset_key = f"{config.date}_{config.mouse}_{config.run}"
    progress = ProgressState(dataset_key)
    
    # Complete M1 successfully
    progress.mark_complete("M1")
    state_manager.save_progress(progress)
    
    # Simulate M2 failure (don't mark complete)
    # Progress should remain unchanged
    
    # Verify M1 still complete, M2 not complete
    loaded_progress = state_manager.load_progress(dataset_key)
    assert loaded_progress.is_complete("M1")
    assert not loaded_progress.is_complete("M2")
    
    # Retry M2 successfully
    progress.mark_complete("M2")
    state_manager.save_progress(progress)
    
    # Verify both complete
    loaded_progress = state_manager.load_progress(dataset_key)
    assert loaded_progress.is_complete("M1")
    assert loaded_progress.is_complete("M2")


def test_list_datasets_workflow(temp_state_dir):
    """
    Test listing all datasets with saved progress.
    """
    state_manager = StateManager(temp_state_dir)
    
    # Create progress for multiple datasets
    datasets = [
        "2025-01-01_mouse1_run1",
        "2025-01-02_mouse2_run2",
        "2025-01-03_mouse3_run3"
    ]
    
    for dataset_key in datasets:
        progress = ProgressState(dataset_key)
        progress.mark_complete("M1")
        state_manager.save_progress(progress)
    
    # List datasets
    saved_datasets = state_manager.list_datasets()
    
    # Verify all datasets are listed
    for dataset_key in datasets:
        assert dataset_key in saved_datasets, \
            f"Dataset {dataset_key} should be in saved datasets"
