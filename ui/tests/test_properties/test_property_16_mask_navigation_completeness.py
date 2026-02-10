"""
Property 16: Mask Navigation Completeness

For any dataset with N masks, the navigation controls should allow reaching
all N masks through sequential next/previous operations.

Validates: Requirements 5.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np
from pathlib import Path
import tempfile
import tifffile

from ui.views.curation_window import CurationWindow


# Feature: pipeline-ui, Property 16: Mask Navigation Completeness
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=1, max_value=20)
)
def test_navigation_reaches_all_masks_forward(qtbot, num_masks):
    """Property: Sequential 'next' operations should reach all N masks.
    
    This property verifies that:
    1. Starting from mask 0, clicking 'next' N-1 times reaches mask N-1
    2. All intermediate masks are visited
    3. Navigation is sequential and complete
    """
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
        
        # Verify all masks were loaded
        assert len(window.masks) == num_masks
        
        # Start at mask 0
        assert window.current_mask_index == 0
        
        visited_indices = [window.current_mask_index]
        
        # Navigate forward through all masks
        for i in range(num_masks - 1):
            window.on_next()
            visited_indices.append(window.current_mask_index)
        
        # Property: All masks were visited
        assert len(visited_indices) == num_masks
        
        # Property: Visited indices are sequential
        assert visited_indices == list(range(num_masks))
        
        # Property: Final position is last mask
        assert window.current_mask_index == num_masks - 1


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=2, max_value=20)
)
def test_navigation_reaches_all_masks_backward(qtbot, num_masks):
    """Property: Sequential 'previous' operations should reach all N masks in reverse.
    
    This property verifies that:
    1. Starting from mask N-1, clicking 'previous' N-1 times reaches mask 0
    2. All intermediate masks are visited in reverse order
    3. Backward navigation is complete
    """
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
        
        # Navigate to last mask
        window.display_mask(num_masks - 1)
        assert window.current_mask_index == num_masks - 1
        
        visited_indices = [window.current_mask_index]
        
        # Navigate backward through all masks
        for i in range(num_masks - 1):
            window.on_previous()
            visited_indices.append(window.current_mask_index)
        
        # Property: All masks were visited
        assert len(visited_indices) == num_masks
        
        # Property: Visited indices are reverse sequential
        assert visited_indices == list(range(num_masks - 1, -1, -1))
        
        # Property: Final position is first mask
        assert window.current_mask_index == 0


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=2, max_value=20),
    target_index=st.integers(min_value=0, max_value=19)
)
def test_navigation_can_reach_any_mask(qtbot, num_masks, target_index):
    """Property: Any mask can be reached through navigation.
    
    This property verifies that:
    1. Starting from mask 0, any target mask can be reached
    2. The path is deterministic (always takes the same number of steps)
    """
    # Ensure target_index is valid
    assume(target_index < num_masks)
    
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
        
        # Start at mask 0
        assert window.current_mask_index == 0
        
        # Navigate to target mask
        for _ in range(target_index):
            window.on_next()
        
        # Property: Target mask was reached
        assert window.current_mask_index == target_index


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=1, max_value=20)
)
def test_navigation_buttons_enabled_correctly(qtbot, num_masks):
    """Property: Navigation buttons should be enabled/disabled correctly.
    
    This property verifies that:
    1. At first mask, previous button is disabled
    2. At last mask, next button is disabled
    3. In middle, both buttons are enabled
    """
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
        
        # At first mask
        window.display_mask(0)
        
        # Property: Previous button disabled at first mask
        assert not window.prev_button.isEnabled()
        
        if num_masks > 1:
            # Property: Next button enabled when not at last mask
            assert window.next_button.isEnabled()
            
            # Navigate to middle (if exists)
            if num_masks > 2:
                middle = num_masks // 2
                window.display_mask(middle)
                
                # Property: Both buttons enabled in middle
                assert window.prev_button.isEnabled()
                assert window.next_button.isEnabled()
            
            # Navigate to last mask
            window.display_mask(num_masks - 1)
            
            # Property: Next button disabled at last mask
            assert not window.next_button.isEnabled()
            
            # Property: Previous button enabled when not at first mask
            assert window.prev_button.isEnabled()
        else:
            # Property: With only one mask, next button is disabled
            assert not window.next_button.isEnabled()


@pytest.mark.property
def test_navigation_with_no_masks(qtbot):
    """Property: Navigation should handle empty mask list gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create empty labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Property: No masks loaded
        assert len(window.masks) == 0
        
        # Property: Navigation operations don't crash
        window.on_next()
        window.on_previous()
        
        # Property: Current index remains 0
        assert window.current_mask_index == 0
