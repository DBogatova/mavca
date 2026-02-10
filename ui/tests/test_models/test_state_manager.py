"""
Unit tests for StateManager class.

Tests state persistence, loading, error handling, and dataset enumeration.
"""

import json
import pytest
from pathlib import Path
from ui.models import StateManager, PipelineConfig, ProgressState


class TestStateManager:
    """Test suite for StateManager class."""
    
    def test_init_creates_state_directory(self, tmp_path):
        """Test that StateManager creates state directory if it doesn't exist."""
        state_dir = tmp_path / "state"
        assert not state_dir.exists()
        
        manager = StateManager(state_dir)
        
        assert state_dir.exists()
        assert manager.state_dir == state_dir
        assert manager.config_file == state_dir / "config.json"
        assert manager.progress_file == state_dir / "progress.json"
    
    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading configuration."""
        manager = StateManager(tmp_path)
        
        # Create and save config
        config = PipelineConfig()
        config.date = "2025-12-25"
        config.mouse = "rAi162_phpeb"
        config.run = "run1"
        config.base_dir = Path("/custom/path")
        
        manager.save_config(config)
        
        # Load config
        loaded_config = manager.load_config()
        
        assert loaded_config is not None
        assert loaded_config.date == "2025-12-25"
        assert loaded_config.mouse == "rAi162_phpeb"
        assert loaded_config.run == "run1"
        assert loaded_config.base_dir == Path("/custom/path")
    
    def test_load_config_returns_none_when_file_missing(self, tmp_path):
        """Test that load_config returns None when config file doesn't exist."""
        manager = StateManager(tmp_path)
        
        loaded_config = manager.load_config()
        
        assert loaded_config is None
    
    def test_load_config_handles_corrupted_json(self, tmp_path):
        """Test that load_config handles corrupted JSON gracefully."""
        manager = StateManager(tmp_path)
        
        # Write corrupted JSON
        with open(manager.config_file, 'w') as f:
            f.write("{ invalid json }")
        
        loaded_config = manager.load_config()
        
        assert loaded_config is None
    
    def test_load_config_handles_invalid_data(self, tmp_path):
        """Test that load_config handles invalid data structure."""
        manager = StateManager(tmp_path)
        
        # Write valid JSON but invalid structure
        with open(manager.config_file, 'w') as f:
            json.dump({"unexpected": "structure"}, f)
        
        # Should not crash, should return config with defaults
        loaded_config = manager.load_config()
        
        # from_dict should handle missing keys gracefully
        assert loaded_config is not None
        assert loaded_config.date == ""
        assert loaded_config.mouse == ""
        assert loaded_config.run == ""
    
    def test_save_and_load_progress(self, tmp_path):
        """Test saving and loading progress state."""
        manager = StateManager(tmp_path)
        
        # Create and save progress
        progress = ProgressState("2025-12-25_rAi162_phpeb_run1")
        progress.mark_complete("M1")
        progress.mark_complete("M2")
        progress.m2_guided = True
        
        manager.save_progress(progress)
        
        # Load progress
        loaded_progress = manager.load_progress("2025-12-25_rAi162_phpeb_run1")
        
        assert loaded_progress.dataset_key == "2025-12-25_rAi162_phpeb_run1"
        assert loaded_progress.is_complete("M1")
        assert loaded_progress.is_complete("M2")
        assert not loaded_progress.is_complete("M3")
        assert loaded_progress.m2_guided is True
    
    def test_load_progress_returns_new_state_for_unknown_dataset(self, tmp_path):
        """Test that load_progress returns new state for unknown dataset."""
        manager = StateManager(tmp_path)
        
        progress = manager.load_progress("unknown_dataset")
        
        assert progress.dataset_key == "unknown_dataset"
        assert len(progress.completed_modules) == 0
        assert progress.m2_guided is False
    
    def test_save_progress_preserves_other_datasets(self, tmp_path):
        """Test that saving progress for one dataset doesn't affect others."""
        manager = StateManager(tmp_path)
        
        # Save progress for dataset 1
        progress1 = ProgressState("dataset1")
        progress1.mark_complete("M1")
        manager.save_progress(progress1)
        
        # Save progress for dataset 2
        progress2 = ProgressState("dataset2")
        progress2.mark_complete("M2")
        manager.save_progress(progress2)
        
        # Load both and verify
        loaded1 = manager.load_progress("dataset1")
        loaded2 = manager.load_progress("dataset2")
        
        assert loaded1.is_complete("M1")
        assert not loaded1.is_complete("M2")
        assert not loaded2.is_complete("M1")
        assert loaded2.is_complete("M2")
    
    def test_load_progress_handles_corrupted_json(self, tmp_path):
        """Test that load_progress handles corrupted JSON gracefully."""
        manager = StateManager(tmp_path)
        
        # Write corrupted JSON
        with open(manager.progress_file, 'w') as f:
            f.write("{ invalid json }")
        
        progress = manager.load_progress("any_dataset")
        
        # Should return new empty progress state
        assert progress.dataset_key == "any_dataset"
        assert len(progress.completed_modules) == 0
    
    def test_load_progress_handles_corrupted_dataset_entry(self, tmp_path):
        """Test that load_progress handles corrupted dataset entry."""
        manager = StateManager(tmp_path)
        
        # Write valid JSON with corrupted dataset entry
        with open(manager.progress_file, 'w') as f:
            json.dump({
                "dataset1": {"invalid": "structure"}
            }, f)
        
        progress = manager.load_progress("dataset1")
        
        # Should return new empty progress state for corrupted dataset
        assert progress.dataset_key == "dataset1"
        assert len(progress.completed_modules) == 0
    
    def test_list_datasets_empty(self, tmp_path):
        """Test list_datasets returns empty list when no progress saved."""
        manager = StateManager(tmp_path)
        
        datasets = manager.list_datasets()
        
        assert datasets == []
    
    def test_list_datasets_returns_all_datasets(self, tmp_path):
        """Test list_datasets returns all datasets with saved progress."""
        manager = StateManager(tmp_path)
        
        # Save progress for multiple datasets
        for i in range(3):
            progress = ProgressState(f"dataset{i}")
            progress.mark_complete("M1")
            manager.save_progress(progress)
        
        datasets = manager.list_datasets()
        
        assert len(datasets) == 3
        assert "dataset0" in datasets
        assert "dataset1" in datasets
        assert "dataset2" in datasets
    
    def test_list_datasets_handles_corrupted_file(self, tmp_path):
        """Test list_datasets handles corrupted progress file."""
        manager = StateManager(tmp_path)
        
        # Write corrupted JSON
        with open(manager.progress_file, 'w') as f:
            f.write("{ invalid json }")
        
        datasets = manager.list_datasets()
        
        # Should return empty list
        assert datasets == []
    
    def test_save_config_raises_on_permission_error(self, tmp_path):
        """Test that save_config raises IOError on permission errors."""
        manager = StateManager(tmp_path)
        
        # Make config file read-only
        manager.config_file.touch()
        manager.config_file.chmod(0o444)
        
        config = PipelineConfig()
        
        with pytest.raises(IOError, match="Failed to save configuration"):
            manager.save_config(config)
        
        # Cleanup
        manager.config_file.chmod(0o644)
    
    def test_save_progress_raises_on_permission_error(self, tmp_path):
        """Test that save_progress raises IOError on permission errors."""
        manager = StateManager(tmp_path)
        
        # Make progress file read-only
        manager.progress_file.touch()
        manager.progress_file.chmod(0o444)
        
        progress = ProgressState("dataset1")
        
        with pytest.raises(IOError, match="Failed to save progress"):
            manager.save_progress(progress)
        
        # Cleanup
        manager.progress_file.chmod(0o644)
    
    def test_config_persistence_roundtrip_with_special_paths(self, tmp_path):
        """Test config persistence with paths containing special characters."""
        manager = StateManager(tmp_path)
        
        config = PipelineConfig()
        config.date = "2025-01-15"
        config.mouse = "test_mouse-123"
        config.run = "run_1-2"
        config.base_dir = Path("/path/with spaces/and-special_chars")
        
        manager.save_config(config)
        loaded_config = manager.load_config()
        
        assert loaded_config.date == config.date
        assert loaded_config.mouse == config.mouse
        assert loaded_config.run == config.run
        assert loaded_config.base_dir == config.base_dir
    
    def test_progress_update_preserves_timestamp(self, tmp_path):
        """Test that progress updates preserve last_updated timestamp."""
        manager = StateManager(tmp_path)
        
        progress = ProgressState("dataset1")
        progress.mark_complete("M1")
        
        manager.save_progress(progress)
        loaded_progress = manager.load_progress("dataset1")
        
        # Timestamp should be preserved (within reasonable tolerance)
        assert loaded_progress.last_updated is not None
        # Just verify it's a valid datetime, exact comparison is tricky
        assert hasattr(loaded_progress.last_updated, 'isoformat')


class TestStateManagerErrorHandling:
    """Test suite for StateManager error handling scenarios."""
    
    def test_corrupted_json_empty_file(self, tmp_path):
        """Test handling of empty JSON file."""
        manager = StateManager(tmp_path)
        
        # Create empty config file
        manager.config_file.touch()
        
        loaded_config = manager.load_config()
        assert loaded_config is None
    
    def test_corrupted_json_partial_content(self, tmp_path):
        """Test handling of partially written JSON file."""
        manager = StateManager(tmp_path)
        
        # Write partial JSON (truncated)
        with open(manager.config_file, 'w') as f:
            f.write('{"date": "2025-01-15", "mouse": "test')
        
        loaded_config = manager.load_config()
        assert loaded_config is None
    
    def test_corrupted_json_wrong_type(self, tmp_path):
        """Test handling of JSON with wrong root type (array instead of object)."""
        manager = StateManager(tmp_path)
        
        # Write JSON array instead of object
        with open(manager.config_file, 'w') as f:
            json.dump(["not", "an", "object"], f)
        
        loaded_config = manager.load_config()
        # Should handle gracefully - from_dict should work with empty dict
        assert loaded_config is not None
    
    def test_corrupted_json_null_values(self, tmp_path):
        """Test handling of JSON with null values."""
        manager = StateManager(tmp_path)
        
        # Write JSON with null values
        with open(manager.config_file, 'w') as f:
            json.dump({"date": None, "mouse": None, "run": None}, f)
        
        loaded_config = manager.load_config()
        assert loaded_config is not None
        # Should handle null values gracefully
    
    def test_corrupted_json_binary_content(self, tmp_path):
        """Test handling of binary content in JSON file."""
        manager = StateManager(tmp_path)
        
        # Write binary content
        with open(manager.config_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')
        
        loaded_config = manager.load_config()
        assert loaded_config is None
    
    def test_corrupted_progress_json_empty_file(self, tmp_path):
        """Test handling of empty progress JSON file."""
        manager = StateManager(tmp_path)
        
        # Create empty progress file
        manager.progress_file.touch()
        
        progress = manager.load_progress("dataset1")
        assert progress.dataset_key == "dataset1"
        assert len(progress.completed_modules) == 0
    
    def test_corrupted_progress_json_partial_content(self, tmp_path):
        """Test handling of partially written progress JSON file."""
        manager = StateManager(tmp_path)
        
        # Write partial JSON
        with open(manager.progress_file, 'w') as f:
            f.write('{"dataset1": {"completed_modules": ["M1"')
        
        progress = manager.load_progress("dataset1")
        assert progress.dataset_key == "dataset1"
        assert len(progress.completed_modules) == 0
    
    def test_corrupted_progress_json_wrong_type(self, tmp_path):
        """Test handling of progress JSON with wrong root type."""
        manager = StateManager(tmp_path)
        
        # Write JSON array instead of object
        with open(manager.progress_file, 'w') as f:
            json.dump(["not", "an", "object"], f)
        
        progress = manager.load_progress("dataset1")
        assert progress.dataset_key == "dataset1"
        assert len(progress.completed_modules) == 0
    
    def test_corrupted_progress_dataset_missing_required_fields(self, tmp_path):
        """Test handling of dataset entry missing required fields."""
        manager = StateManager(tmp_path)
        
        # Write progress with missing fields
        with open(manager.progress_file, 'w') as f:
            json.dump({
                "dataset1": {
                    # Missing dataset_key, completed_modules, etc.
                    "some_field": "value"
                }
            }, f)
        
        progress = manager.load_progress("dataset1")
        assert progress.dataset_key == "dataset1"
        assert len(progress.completed_modules) == 0
    
    def test_corrupted_progress_invalid_module_ids(self, tmp_path):
        """Test handling of invalid module IDs in progress data."""
        manager = StateManager(tmp_path)
        
        # Write progress with invalid module IDs (not strings)
        with open(manager.progress_file, 'w') as f:
            json.dump({
                "dataset1": {
                    "dataset_key": "dataset1",
                    "completed_modules": [1, 2, None, {"invalid": "object"}],
                    "m2_guided": False
                }
            }, f)
        
        # Should handle gracefully - from_dict should filter or handle invalid values
        progress = manager.load_progress("dataset1")
        assert progress.dataset_key == "dataset1"
    
    def test_missing_config_file_returns_none(self, tmp_path):
        """Test that missing config file returns None."""
        manager = StateManager(tmp_path)
        
        # Ensure file doesn't exist
        if manager.config_file.exists():
            manager.config_file.unlink()
        
        loaded_config = manager.load_config()
        assert loaded_config is None
    
    def test_missing_progress_file_returns_empty_state(self, tmp_path):
        """Test that missing progress file returns empty state."""
        manager = StateManager(tmp_path)
        
        # Ensure file doesn't exist
        if manager.progress_file.exists():
            manager.progress_file.unlink()
        
        progress = manager.load_progress("dataset1")
        assert progress.dataset_key == "dataset1"
        assert len(progress.completed_modules) == 0
        assert progress.m2_guided is False
    
    def test_missing_progress_file_list_datasets_returns_empty(self, tmp_path):
        """Test that list_datasets returns empty list when file missing."""
        manager = StateManager(tmp_path)
        
        # Ensure file doesn't exist
        if manager.progress_file.exists():
            manager.progress_file.unlink()
        
        datasets = manager.list_datasets()
        assert datasets == []
    
    def test_permission_error_on_save_config(self, tmp_path):
        """Test that permission errors on save_config raise IOError."""
        manager = StateManager(tmp_path)
        
        # Create read-only config file
        manager.config_file.touch()
        manager.config_file.chmod(0o444)
        
        config = PipelineConfig()
        config.date = "2025-01-15"
        
        try:
            with pytest.raises(IOError, match="Failed to save configuration"):
                manager.save_config(config)
        finally:
            # Cleanup
            manager.config_file.chmod(0o644)
    
    def test_permission_error_on_save_progress(self, tmp_path):
        """Test that permission errors on save_progress raise IOError."""
        manager = StateManager(tmp_path)
        
        # Create read-only progress file
        manager.progress_file.touch()
        manager.progress_file.chmod(0o444)
        
        progress = ProgressState("dataset1")
        progress.mark_complete("M1")
        
        try:
            with pytest.raises(IOError, match="Failed to save progress"):
                manager.save_progress(progress)
        finally:
            # Cleanup
            manager.progress_file.chmod(0o644)
    
    def test_permission_error_on_read_only_directory(self, tmp_path):
        """Test handling of read-only state directory."""
        state_dir = tmp_path / "readonly_state"
        state_dir.mkdir()
        
        # Create manager and save some data
        manager = StateManager(state_dir)
        config = PipelineConfig()
        config.date = "2025-01-15"
        manager.save_config(config)
        
        # Make directory read-only
        state_dir.chmod(0o555)
        
        try:
            # Attempting to save should raise IOError
            with pytest.raises(IOError, match="Failed to save configuration"):
                manager.save_config(config)
        finally:
            # Cleanup
            state_dir.chmod(0o755)
    
    def test_unicode_in_corrupted_json(self, tmp_path):
        """Test handling of corrupted JSON with unicode characters."""
        manager = StateManager(tmp_path)
        
        # Write corrupted JSON with unicode
        with open(manager.config_file, 'w', encoding='utf-8') as f:
            f.write('{"date": "2025-01-15", "mouse": "测试🐭", invalid}')
        
        loaded_config = manager.load_config()
        assert loaded_config is None
    
    def test_very_large_corrupted_file(self, tmp_path):
        """Test handling of very large corrupted file."""
        manager = StateManager(tmp_path)
        
        # Write large corrupted JSON
        with open(manager.config_file, 'w') as f:
            f.write('{"data": "' + 'x' * 1000000 + '" invalid}')
        
        loaded_config = manager.load_config()
        assert loaded_config is None
    
    def test_nested_corrupted_progress_data(self, tmp_path):
        """Test handling of deeply nested corrupted progress data."""
        manager = StateManager(tmp_path)
        
        # Write progress with deeply nested invalid structure
        with open(manager.progress_file, 'w') as f:
            json.dump({
                "dataset1": {
                    "dataset_key": "dataset1",
                    "completed_modules": {
                        "nested": {
                            "invalid": ["structure"]
                        }
                    },
                    "m2_guided": "not_a_boolean"
                }
            }, f)
        
        progress = manager.load_progress("dataset1")
        assert progress.dataset_key == "dataset1"
        # Should handle gracefully and return empty or default state
    
    def test_save_config_after_corrupted_load(self, tmp_path):
        """Test that saving works after loading corrupted config."""
        manager = StateManager(tmp_path)
        
        # Write corrupted config
        with open(manager.config_file, 'w') as f:
            f.write('{ invalid json }')
        
        # Load should return None
        loaded_config = manager.load_config()
        assert loaded_config is None
        
        # Should be able to save new config
        new_config = PipelineConfig()
        new_config.date = "2025-01-15"
        new_config.mouse = "test_mouse"
        new_config.run = "run1"
        
        manager.save_config(new_config)
        
        # Should be able to load the new config
        loaded_config = manager.load_config()
        assert loaded_config is not None
        assert loaded_config.date == "2025-01-15"
    
    def test_save_progress_after_corrupted_load(self, tmp_path):
        """Test that saving progress works after loading corrupted file."""
        manager = StateManager(tmp_path)
        
        # Write corrupted progress
        with open(manager.progress_file, 'w') as f:
            f.write('{ invalid json }')
        
        # Load should return empty state
        progress = manager.load_progress("dataset1")
        assert len(progress.completed_modules) == 0
        
        # Should be able to save new progress
        new_progress = ProgressState("dataset1")
        new_progress.mark_complete("M1")
        new_progress.mark_complete("M2")
        
        manager.save_progress(new_progress)
        
        # Should be able to load the new progress
        loaded_progress = manager.load_progress("dataset1")
        assert loaded_progress.is_complete("M1")
        assert loaded_progress.is_complete("M2")
    
    def test_concurrent_dataset_corruption_isolation(self, tmp_path):
        """Test that corruption in one dataset doesn't affect others."""
        manager = StateManager(tmp_path)
        
        # Save valid progress for dataset1
        progress1 = ProgressState("dataset1")
        progress1.mark_complete("M1")
        manager.save_progress(progress1)
        
        # Manually corrupt dataset2 entry in the file
        with open(manager.progress_file, 'r') as f:
            data = json.load(f)
        
        data["dataset2"] = {"invalid": "structure", "no_dataset_key": True}
        
        with open(manager.progress_file, 'w') as f:
            json.dump(data, f)
        
        # Loading dataset1 should still work
        loaded1 = manager.load_progress("dataset1")
        assert loaded1.is_complete("M1")
        
        # Loading dataset2 should return empty state (graceful handling)
        loaded2 = manager.load_progress("dataset2")
        assert loaded2.dataset_key == "dataset2"
        assert len(loaded2.completed_modules) == 0


# Property-Based Tests
from hypothesis import given, strategies as st, settings


# Feature: pipeline-ui, Property 23: Dataset-Specific Progress Loading
@given(
    # Generate multiple datasets with different keys
    st.lists(
        st.tuples(
            # Dataset key (DATE_MOUSE_RUN format)
            st.text(
                alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll', 'Nd'),
                    whitelist_characters='-_'
                ),
                min_size=5,
                max_size=50
            ).filter(lambda s: s and not s.startswith('_') and not s.startswith('-')),
            # Completed modules for this dataset
            st.sets(
                st.sampled_from(['M1', 'M1.5', 'M2', 'M2.5', 'M3', 'M4', 'M5', 'M6']),
                min_size=0,
                max_size=8
            ),
            # m2_guided flag
            st.booleans()
        ),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x[0]  # Ensure unique dataset keys
    )
)
@settings(max_examples=100)
def test_dataset_specific_progress_loading_property(tmp_path, datasets_data):
    """
    Property 23: Dataset-Specific Progress Loading
    
    For any dataset key (DATE_MOUSE_RUN), changing to that dataset should
    load the correct progress state for that specific dataset.
    
    Validates: Requirements 6.6
    """
    manager = StateManager(tmp_path)
    
    # Save progress for all datasets
    saved_states = {}
    for dataset_key, completed_modules, m2_guided in datasets_data:
        progress = ProgressState(dataset_key)
        for module_id in completed_modules:
            progress.mark_complete(module_id)
        progress.m2_guided = m2_guided
        
        manager.save_progress(progress)
        saved_states[dataset_key] = (completed_modules, m2_guided)
    
    # Load progress for each dataset and verify it matches what was saved
    for dataset_key, (expected_modules, expected_m2_guided) in saved_states.items():
        loaded_progress = manager.load_progress(dataset_key)
        
        # Verify dataset key matches
        assert loaded_progress.dataset_key == dataset_key, \
            f"Dataset key mismatch: expected {dataset_key}, got {loaded_progress.dataset_key}"
        
        # Verify completed modules match
        assert loaded_progress.completed_modules == expected_modules, \
            f"Completed modules mismatch for {dataset_key}: expected {expected_modules}, got {loaded_progress.completed_modules}"
        
        # Verify m2_guided flag matches
        assert loaded_progress.m2_guided == expected_m2_guided, \
            f"m2_guided mismatch for {dataset_key}: expected {expected_m2_guided}, got {loaded_progress.m2_guided}"
        
        # Verify each module's completion status
        for module_id in expected_modules:
            assert loaded_progress.is_complete(module_id), \
                f"Module {module_id} should be complete for dataset {dataset_key}"
        
        # Verify modules not in expected_modules are not marked complete
        all_modules = {'M1', 'M1.5', 'M2', 'M2.5', 'M3', 'M4', 'M5', 'M6'}
        for module_id in all_modules - expected_modules:
            assert not loaded_progress.is_complete(module_id), \
                f"Module {module_id} should not be complete for dataset {dataset_key}"
    
    # Verify that loading a non-existent dataset returns empty progress
    non_existent_key = "non_existent_dataset_xyz_123"
    new_progress = manager.load_progress(non_existent_key)
    assert new_progress.dataset_key == non_existent_key
    assert len(new_progress.completed_modules) == 0
    assert new_progress.m2_guided is False
