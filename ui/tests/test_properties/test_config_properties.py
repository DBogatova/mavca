"""
Property-based tests for PipelineConfig class.

These tests use Hypothesis to verify universal correctness properties
across all possible inputs.
"""

import re
from hypothesis import given, strategies as st
from pathlib import Path
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


# Feature: pipeline-ui, Property 5: Date Format Validation
@given(st.text())
def test_date_validation_property(date_string):
    """
    For any string input, date validation should accept it
    if and only if it matches YYYY-MM-DD format with valid values.
    
    Validates: Requirements 2.2
    """
    config = PipelineConfig()
    is_valid = config.validate_date(date_string)
    
    # Check if string matches expected format
    expected_valid = _is_valid_date_format(date_string)
    
    assert is_valid == expected_valid, (
        f"Date validation mismatch for '{date_string}': "
        f"got {is_valid}, expected {expected_valid}"
    )


# Feature: pipeline-ui, Property 6: Identifier Validation
@given(st.text())
def test_identifier_validation_property(identifier_string):
    """
    For any string input, identifier validation should accept it
    if and only if it contains only alphanumeric characters, underscores, and hyphens.
    
    Validates: Requirements 2.3, 2.4
    """
    config = PipelineConfig()
    is_valid = config.validate_identifier(identifier_string)
    
    # Check if string matches expected format
    expected_valid = _is_valid_identifier(identifier_string)
    
    assert is_valid == expected_valid, (
        f"Identifier validation mismatch for '{identifier_string}': "
        f"got {is_valid}, expected {expected_valid}"
    )


# Feature: pipeline-ui, Property 7: Configuration Persistence Round-Trip
@given(
    st.text(min_size=10, max_size=10).filter(lambda s: _is_valid_date_format(s)),
    st.text(min_size=1, max_size=50).filter(lambda s: _is_valid_identifier(s)),
    st.text(min_size=1, max_size=50).filter(lambda s: _is_valid_identifier(s)),
    st.text(min_size=1, max_size=100)
)
def test_config_persistence_roundtrip(date, mouse, run, base_dir_str):
    """
    For any valid configuration, saving then loading should
    produce an equivalent configuration.
    
    Validates: Requirements 2.5
    """
    config = PipelineConfig()
    config.date = date
    config.mouse = mouse
    config.run = run
    config.base_dir = Path(base_dir_str)
    
    # Serialize to dict
    data = config.to_dict()
    
    # Deserialize from dict
    loaded_config = PipelineConfig.from_dict(data)
    
    # Verify all fields match
    assert loaded_config.date == config.date, "Date mismatch after round-trip"
    assert loaded_config.mouse == config.mouse, "Mouse mismatch after round-trip"
    assert loaded_config.run == config.run, "Run mismatch after round-trip"
    assert loaded_config.base_dir == config.base_dir, "Base dir mismatch after round-trip"


# Feature: pipeline-ui, Property 27: Output Path Construction
@given(
    st.text(min_size=10, max_size=10).filter(lambda s: _is_valid_date_format(s)),
    st.text(min_size=1, max_size=50).filter(lambda s: _is_valid_identifier(s)),
    st.text(min_size=1, max_size=50).filter(lambda s: _is_valid_identifier(s))
)
def test_output_path_construction(date, mouse, run):
    """
    For any valid DATE, MOUSE, and RUN configuration, the constructed
    output paths should follow the pattern data/DATE/MOUSE/RUN/.
    
    Validates: Requirements 7.5, 11.1, 11.2
    """
    config = PipelineConfig()
    config.date = date
    config.mouse = mouse
    config.run = run
    
    # Get constructed path
    data_path = config.get_data_path()
    
    # Verify path structure
    expected_suffix = Path("data") / date / mouse / run
    assert data_path.parts[-4:] == expected_suffix.parts, (
        f"Path construction incorrect: expected suffix {expected_suffix}, "
        f"got {Path(*data_path.parts[-4:])}"
    )
    
    # Verify path components
    assert date in str(data_path), f"Date '{date}' not in path {data_path}"
    assert mouse in str(data_path), f"Mouse '{mouse}' not in path {data_path}"
    assert run in str(data_path), f"Run '{run}' not in path {data_path}"



# Import ProgressState for progress tracking tests
from ui.models.progress_state import ProgressState


# Feature: pipeline-ui, Property 19: Progress State Tracking
@given(
    st.text(min_size=1, max_size=50),
    st.lists(st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]), min_size=0, max_size=8, unique=True)
)
def test_progress_state_tracking(dataset_key, completed_modules):
    """
    For any set of completed modules, the progress state should
    accurately reflect which modules are completed.
    
    Validates: Requirements 6.1
    """
    progress = ProgressState(dataset_key)
    
    # Mark modules as complete
    for module_id in completed_modules:
        progress.mark_complete(module_id)
    
    # Verify all marked modules are complete
    for module_id in completed_modules:
        assert progress.is_complete(module_id), (
            f"Module {module_id} should be marked as complete"
        )
    
    # Verify unmarked modules are not complete
    all_modules = {"M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"}
    unmarked_modules = all_modules - set(completed_modules)
    for module_id in unmarked_modules:
        assert not progress.is_complete(module_id), (
            f"Module {module_id} should not be marked as complete"
        )


# Feature: pipeline-ui, Property 21: Progress Persistence Round-Trip
@given(
    st.text(min_size=1, max_size=50),
    st.lists(st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]), min_size=0, max_size=8, unique=True),
    st.booleans()
)
def test_progress_persistence_roundtrip(dataset_key, completed_modules, m2_guided):
    """
    For any progress state for a dataset, saving then loading should
    produce an equivalent progress state.
    
    Validates: Requirements 6.4
    """
    progress = ProgressState(dataset_key)
    
    # Set up progress state
    for module_id in completed_modules:
        progress.mark_complete(module_id)
    progress.m2_guided = m2_guided
    
    # Serialize to dict
    data = progress.to_dict()
    
    # Deserialize from dict
    loaded_progress = ProgressState.from_dict(data)
    
    # Verify all fields match
    assert loaded_progress.dataset_key == progress.dataset_key, "Dataset key mismatch"
    assert loaded_progress.completed_modules == progress.completed_modules, "Completed modules mismatch"
    assert loaded_progress.m2_guided == progress.m2_guided, "M2 guided flag mismatch"


# Feature: pipeline-ui, Property 22: Progress Reset
@given(
    st.text(min_size=1, max_size=50),
    st.lists(st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]), min_size=1, max_size=8, unique=True)
)
def test_progress_reset(dataset_key, completed_modules):
    """
    For any progress state with completed modules, resetting should
    clear all completions.
    
    Validates: Requirements 6.5
    """
    progress = ProgressState(dataset_key)
    
    # Mark modules as complete
    for module_id in completed_modules:
        progress.mark_complete(module_id)
    progress.m2_guided = True
    
    # Verify modules are complete before reset
    assert len(progress.completed_modules) > 0, "Should have completed modules before reset"
    
    # Reset progress
    progress.reset()
    
    # Verify all completions are cleared
    assert len(progress.completed_modules) == 0, "All completions should be cleared after reset"
    assert progress.m2_guided is False, "M2 guided flag should be reset"
    
    # Verify no modules are marked as complete
    all_modules = {"M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"}
    for module_id in all_modules:
        assert not progress.is_complete(module_id), (
            f"Module {module_id} should not be complete after reset"
        )


# Import StateManager for dataset-specific progress loading test
from ui.models.state_manager import StateManager
import tempfile


# Feature: pipeline-ui, Property 23: Dataset-Specific Progress Loading
@given(
    st.lists(
        st.tuples(
            st.text(min_size=10, max_size=10).filter(lambda s: _is_valid_date_format(s)),
            st.text(min_size=1, max_size=20).filter(lambda s: _is_valid_identifier(s)),
            st.text(min_size=1, max_size=20).filter(lambda s: _is_valid_identifier(s)),
            st.lists(st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]), min_size=0, max_size=8, unique=True),
            st.booleans()
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda x: f"{x[0]}_{x[1]}_{x[2]}"  # Ensure unique dataset keys
    )
)
def test_dataset_specific_progress_loading(datasets_data):
    """
    For any dataset key (DATE_MOUSE_RUN), changing to that dataset should
    load the correct progress state for that specific dataset.
    
    Validates: Requirements 6.6
    """
    # Create temporary state directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_manager = StateManager(Path(tmp_dir))
        
        # Dictionary to track expected progress for each dataset
        expected_progress = {}
        
        # Save progress for multiple different datasets
        for date, mouse, run, completed_modules, m2_guided in datasets_data:
            dataset_key = f"{date}_{mouse}_{run}"
            
            # Create progress state
            progress = ProgressState(dataset_key)
            for module_id in completed_modules:
                progress.mark_complete(module_id)
            progress.m2_guided = m2_guided
            
            # Save progress
            state_manager.save_progress(progress)
            
            # Store expected state
            expected_progress[dataset_key] = {
                'completed_modules': set(completed_modules),
                'm2_guided': m2_guided
            }
        
        # Verify that loading each dataset key returns the correct progress state
        for dataset_key, expected in expected_progress.items():
            loaded_progress = state_manager.load_progress(dataset_key)
            
            assert loaded_progress.dataset_key == dataset_key, (
                f"Dataset key mismatch: expected {dataset_key}, got {loaded_progress.dataset_key}"
            )
            
            assert loaded_progress.completed_modules == expected['completed_modules'], (
                f"Completed modules mismatch for {dataset_key}: "
                f"expected {expected['completed_modules']}, got {loaded_progress.completed_modules}"
            )
            
            assert loaded_progress.m2_guided == expected['m2_guided'], (
                f"M2 guided flag mismatch for {dataset_key}: "
                f"expected {expected['m2_guided']}, got {loaded_progress.m2_guided}"
            )
            
            # Verify that each dataset's progress is independent
            # (changing one dataset shouldn't affect others)
            for other_key, other_expected in expected_progress.items():
                if other_key != dataset_key:
                    other_progress = state_manager.load_progress(other_key)
                    assert other_progress.completed_modules == other_expected['completed_modules'], (
                        f"Loading {dataset_key} affected {other_key}'s progress"
                    )
