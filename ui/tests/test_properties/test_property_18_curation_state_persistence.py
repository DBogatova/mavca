"""
Property 18: Curation State Persistence Round-Trip

For any curation session with operations performed, exiting and resuming
should preserve all curation state (which masks were kept, deleted, or merged).

Validates: Requirements 5.8
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from pathlib import Path
import tempfile
import tifffile
import json

from ui.views.curation_window import CurationWindow


# Feature: pipeline-ui, Property 18: Curation State Persistence Round-Trip
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=1, max_value=10),
    operations=st.lists(
        st.tuples(
            st.sampled_from(['keep', 'delete']),
            st.integers(min_value=0, max_value=9)
        ),
        min_size=1,
        max_size=5
    )
)
def test_curation_state_persists_after_save(qtbot, num_masks, operations):
    """Property: Saved curation state should be loadable and match original state.
    
    This property verifies that:
    1. Curation operations can be saved to disk
    2. Saved state contains all mask statuses
    3. Saved state can be loaded back
    4. Loaded state matches original state
    """
    # Filter operations to valid indices
    operations = [(op, idx) for op, idx in operations if idx < num_masks]
    assume(len(operations) > 0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create N test masks
        for i in range(num_masks):
            mask_array = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8)
            mask_file = labelmaps_dir / f"labelmap_{i:03d}.tif"
            tifffile.imwrite(mask_file, mask_array)
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Perform operations
        for operation, mask_index in operations:
            window.display_mask(mask_index)
            
            if operation == 'keep':
                window.on_keep()
            elif operation == 'delete':
                window.on_delete()
        
        # Save original state
        original_states = {i: window.masks[i].status for i in range(num_masks)}
        
        # Save curation results
        window.save_curation_results()
        
        # Property: Results file was created
        results_file = data_path / "curation_results.json"
        assert results_file.exists()
        
        # Load saved results
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        # Property: Saved results contain all masks
        assert saved_results["total_masks"] == num_masks
        assert len(saved_results["masks"]) == num_masks
        
        # Property: Each mask's state was saved correctly
        for i in range(num_masks):
            saved_mask = saved_results["masks"][i]
            assert saved_mask["mask_id"] == i
            assert saved_mask["status"] == original_states[i]


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=2, max_value=10),
    source_index=st.integers(min_value=0, max_value=9),
    target_index=st.integers(min_value=0, max_value=9)
)
def test_merge_state_persists_after_save(qtbot, num_masks, source_index, target_index):
    """Property: Merge relationships should be saved and loadable.
    
    This property verifies that:
    1. Merge operations are saved with relationship information
    2. Both masks in a merge are marked as merged
    3. Merge relationship (merged_with) is preserved
    """
    # Ensure indices are valid and different
    assume(source_index < num_masks)
    assume(target_index < num_masks)
    assume(source_index != target_index)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create N test masks
        for i in range(num_masks):
            mask_array = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8)
            mask_file = labelmaps_dir / f"labelmap_{i:03d}.tif"
            tifffile.imwrite(mask_file, mask_array)
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Perform merge operation
        window.display_mask(source_index)
        window.on_merge()
        window.display_mask(target_index)
        window.on_merge()
        
        # Save curation results
        window.save_curation_results()
        
        # Load saved results
        results_file = data_path / "curation_results.json"
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        # Property: Both masks are marked as merged
        source_mask = saved_results["masks"][source_index]
        target_mask = saved_results["masks"][target_index]
        
        assert source_mask["status"] == "merged"
        assert target_mask["status"] == "merged"
        
        # Property: Merge relationship is preserved
        assert source_mask["merged_with"] == target_index
        assert target_mask["merged_with"] == source_index


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    num_masks=st.integers(min_value=3, max_value=8),
    kept_indices=st.lists(st.integers(min_value=0, max_value=7), min_size=0, max_size=3),
    deleted_indices=st.lists(st.integers(min_value=0, max_value=7), min_size=0, max_size=3)
)
def test_complex_curation_state_round_trip(qtbot, num_masks, kept_indices, deleted_indices):
    """Property: Complex curation state with multiple operations should persist correctly.
    
    This property verifies that:
    1. Multiple different operations can be saved
    2. State is preserved accurately for all masks
    3. Statistics (kept, deleted, pending counts) are correct
    """
    # Filter to valid indices and ensure no overlap
    kept_indices = [i for i in kept_indices if i < num_masks]
    deleted_indices = [i for i in deleted_indices if i < num_masks and i not in kept_indices]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create N test masks
        for i in range(num_masks):
            mask_array = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8)
            mask_file = labelmaps_dir / f"labelmap_{i:03d}.tif"
            tifffile.imwrite(mask_file, mask_array)
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Mark masks as kept
        for idx in kept_indices:
            window.display_mask(idx)
            window.on_keep()
        
        # Mark masks as deleted
        for idx in deleted_indices:
            window.display_mask(idx)
            window.on_delete()
        
        # Calculate expected counts
        expected_kept = len(kept_indices)
        expected_deleted = len(deleted_indices)
        expected_pending = num_masks - expected_kept - expected_deleted
        
        # Save curation results
        window.save_curation_results()
        
        # Load saved results
        results_file = data_path / "curation_results.json"
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        # Count saved states
        saved_kept = sum(1 for m in saved_results["masks"] if m["status"] == "kept")
        saved_deleted = sum(1 for m in saved_results["masks"] if m["status"] == "deleted")
        saved_pending = sum(1 for m in saved_results["masks"] if m["status"] == "pending")
        
        # Property: Counts match expected values
        assert saved_kept == expected_kept
        assert saved_deleted == expected_deleted
        assert saved_pending == expected_pending
        
        # Property: Specific masks have correct states
        for idx in kept_indices:
            assert saved_results["masks"][idx]["status"] == "kept"
        
        for idx in deleted_indices:
            assert saved_results["masks"][idx]["status"] == "deleted"


@pytest.mark.property
def test_empty_curation_state_persists(qtbot):
    """Property: Curation state with no operations should still be saveable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create test masks
        for i in range(3):
            mask_array = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8)
            mask_file = labelmaps_dir / f"labelmap_{i:03d}.tif"
            tifffile.imwrite(mask_file, mask_array)
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Don't perform any operations
        
        # Save curation results
        window.save_curation_results()
        
        # Property: Results file was created
        results_file = data_path / "curation_results.json"
        assert results_file.exists()
        
        # Load saved results
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        # Property: All masks are pending
        assert all(m["status"] == "pending" for m in saved_results["masks"])


@pytest.mark.property
def test_save_creates_valid_json(qtbot):
    """Property: Saved curation results should be valid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create test mask
        mask_array = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8)
        mask_file = labelmaps_dir / "labelmap_000.tif"
        tifffile.imwrite(mask_file, mask_array)
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Perform operation
        window.on_keep()
        
        # Save curation results
        window.save_curation_results()
        
        # Property: File can be loaded as JSON
        results_file = data_path / "curation_results.json"
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            json_valid = True
        except json.JSONDecodeError:
            json_valid = False
        
        assert json_valid
        
        # Property: JSON has expected structure
        assert "total_masks" in data
        assert "masks" in data
        assert isinstance(data["masks"], list)
