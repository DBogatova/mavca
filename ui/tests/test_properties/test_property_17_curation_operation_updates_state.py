"""
Property 17: Curation Operation Updates State

For any curation operation (Keep, Delete, Merge), the mask state should be
immediately updated to reflect the operation.

Validates: Requirements 5.6
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from pathlib import Path
import tempfile
import tifffile

from ui.views.curation_window import CurationWindow


# Feature: pipeline-ui, Property 17: Curation Operation Updates State
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=1, max_value=10),
    mask_index=st.integers(min_value=0, max_value=9)
)
def test_keep_operation_updates_state(qtbot, num_masks, mask_index):
    """Property: Keep operation should immediately update mask status to 'kept'.
    
    This property verifies that:
    1. Before keep operation, mask status is 'pending'
    2. After keep operation, mask status is 'kept'
    3. State change is immediate
    """
    # Ensure mask_index is valid
    assume(mask_index < num_masks)
    
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
        
        # Navigate to target mask
        window.display_mask(mask_index)
        
        # Property: Initial status is 'pending'
        assert window.masks[mask_index].status == "pending"
        
        # Perform keep operation
        window.on_keep()
        
        # Property: Status is immediately updated to 'kept'
        assert window.masks[mask_index].status == "kept"


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=1, max_value=10),
    mask_index=st.integers(min_value=0, max_value=9)
)
def test_delete_operation_updates_state(qtbot, num_masks, mask_index):
    """Property: Delete operation should immediately update mask status to 'deleted'.
    
    This property verifies that:
    1. Before delete operation, mask status is 'pending'
    2. After delete operation, mask status is 'deleted'
    3. State change is immediate
    """
    # Ensure mask_index is valid
    assume(mask_index < num_masks)
    
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
        
        # Navigate to target mask
        window.display_mask(mask_index)
        
        # Property: Initial status is 'pending'
        assert window.masks[mask_index].status == "pending"
        
        # Perform delete operation
        window.on_delete()
        
        # Property: Status is immediately updated to 'deleted'
        assert window.masks[mask_index].status == "deleted"


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=2, max_value=10),
    source_index=st.integers(min_value=0, max_value=9),
    target_index=st.integers(min_value=0, max_value=9)
)
def test_merge_operation_updates_state(qtbot, num_masks, source_index, target_index):
    """Property: Merge operation should immediately update both masks' status to 'merged'.
    
    This property verifies that:
    1. Before merge, both masks have status 'pending'
    2. After merge, both masks have status 'merged'
    3. Merge relationship is recorded
    4. State change is immediate
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
        
        # Navigate to source mask
        window.display_mask(source_index)
        
        # Property: Initial status is 'pending'
        assert window.masks[source_index].status == "pending"
        assert window.masks[target_index].status == "pending"
        
        # Start merge operation
        window.on_merge()
        
        # Property: Merge mode is active
        assert window.merge_mode is True
        assert window.merge_source_index == source_index
        
        # Navigate to target mask
        window.display_mask(target_index)
        
        # Complete merge operation
        window.on_merge()
        
        # Property: Both masks are marked as 'merged'
        assert window.masks[source_index].status == "merged"
        assert window.masks[target_index].status == "merged"
        
        # Property: Merge relationship is recorded
        assert window.masks[source_index].merged_with == target_index
        assert window.masks[target_index].merged_with == source_index
        
        # Property: Merge mode is deactivated
        assert window.merge_mode is False
        assert window.merge_source_index is None


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=3, max_value=10),
    operations=st.lists(
        st.tuples(
            st.sampled_from(['keep', 'delete']),
            st.integers(min_value=0, max_value=9)
        ),
        min_size=1,
        max_size=5
    )
)
def test_multiple_operations_update_state_correctly(qtbot, num_masks, operations):
    """Property: Multiple curation operations should each update state correctly.
    
    This property verifies that:
    1. Each operation updates the correct mask
    2. Operations don't interfere with each other
    3. State is consistent after all operations
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
        
        # Track expected states
        expected_states = {i: "pending" for i in range(num_masks)}
        
        # Perform operations
        for operation, mask_index in operations:
            window.display_mask(mask_index)
            
            if operation == 'keep':
                window.on_keep()
                expected_states[mask_index] = "kept"
            elif operation == 'delete':
                window.on_delete()
                expected_states[mask_index] = "deleted"
        
        # Property: All masks have correct final state
        for i in range(num_masks):
            assert window.masks[i].status == expected_states[i]


@pytest.mark.property
def test_operation_on_already_curated_mask(qtbot):
    """Property: Operations on already-curated masks should update state."""
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
        
        # Mark as kept
        window.on_keep()
        assert window.masks[0].status == "kept"
        
        # Change to deleted
        window.display_mask(0)
        window.on_delete()
        
        # Property: Status should be updated to new value
        assert window.masks[0].status == "deleted"
        
        # Change back to kept
        window.display_mask(0)
        window.on_keep()
        
        # Property: Status should be updated again
        assert window.masks[0].status == "kept"
