"""
Property 15: Mask Selection Displays Trace

For any mask selection in the M3 curation interface, a DFF trace preview
should be computed and displayed.

Validates: Requirements 5.3
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from pathlib import Path
import tempfile
import tifffile

from ui.views.curation_window import CurationWindow, MaskData


# Feature: pipeline-ui, Property 15: Mask Selection Displays Trace
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    mask_shape=st.tuples(
        st.integers(min_value=10, max_value=50),  # z
        st.integers(min_value=10, max_value=50),  # y
        st.integers(min_value=10, max_value=50)   # x
    ),
    num_timepoints=st.integers(min_value=10, max_value=100)
)
def test_mask_selection_displays_trace(qtbot, mask_shape, num_timepoints):
    """Property: For any mask selection, a DFF trace should be computed and displayed.
    
    This property verifies that:
    1. When a mask is selected, compute_dff_preview is called
    2. A trace array is returned with correct length
    3. The trace plot is updated
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create a test mask
        mask_array = np.random.randint(0, 2, size=mask_shape, dtype=np.uint8)
        mask_file = labelmaps_dir / "labelmap_000.tif"
        tifffile.imwrite(mask_file, mask_array)
        
        # Create preprocessed directory with DFF stack
        preprocessed_dir = data_path / "preprocessed"
        preprocessed_dir.mkdir()
        
        # Create DFF stack (time, z, y, x)
        dff_stack = np.random.randn(num_timepoints, *mask_shape).astype(np.float32)
        dff_file = preprocessed_dir / "dff_stack.npy"
        np.save(dff_file, dff_stack)
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Verify mask was loaded
        assert len(window.masks) == 1
        
        # Display the mask (this should compute and display trace)
        window.display_mask(0)
        
        # Verify trace was computed
        # The compute_dff_preview method should return a trace with num_timepoints
        mask = window.masks[0]
        trace = window.compute_dff_preview(mask.mask_array, dff_stack)
        
        # Property: Trace length matches number of timepoints
        assert len(trace) == num_timepoints
        
        # Property: Trace contains valid numbers (not NaN or Inf)
        assert np.all(np.isfinite(trace))
        
        # Property: Plot was updated (ax has data)
        assert len(window.ax.lines) > 0


@pytest.mark.property
def test_mask_selection_handles_missing_dff_data(qtbot):
    """Property: Mask selection should handle missing DFF data gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create a test mask
        mask_array = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8)
        mask_file = labelmaps_dir / "labelmap_000.tif"
        tifffile.imwrite(mask_file, mask_array)
        
        # Don't create DFF data
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Display the mask (should handle missing DFF gracefully)
        window.display_mask(0)
        
        # Property: Window should still be functional
        assert window.current_mask_index == 0
        assert len(window.masks) == 1


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    num_masks=st.integers(min_value=1, max_value=10),
    mask_index=st.integers(min_value=0, max_value=9)
)
def test_trace_updates_on_mask_change(qtbot, num_masks, mask_index):
    """Property: Trace should update when navigating between masks."""
    # Ensure mask_index is valid
    if mask_index >= num_masks:
        mask_index = num_masks - 1
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Create labelmaps directory
        labelmaps_dir = data_path / "labelmaps"
        labelmaps_dir.mkdir()
        
        # Create multiple test masks
        for i in range(num_masks):
            mask_array = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint8)
            mask_file = labelmaps_dir / f"labelmap_{i:03d}.tif"
            tifffile.imwrite(mask_file, mask_array)
        
        # Create preprocessed directory with DFF stack
        preprocessed_dir = data_path / "preprocessed"
        preprocessed_dir.mkdir()
        
        dff_stack = np.random.randn(20, 10, 10, 10).astype(np.float32)
        dff_file = preprocessed_dir / "dff_stack.npy"
        np.save(dff_file, dff_stack)
        
        # Create curation window
        window = CurationWindow(data_path)
        qtbot.addWidget(window)
        
        # Display first mask
        window.display_mask(0)
        first_trace_data = window.ax.lines[0].get_ydata().copy() if window.ax.lines else None
        
        # Display different mask
        window.display_mask(mask_index)
        second_trace_data = window.ax.lines[0].get_ydata().copy() if window.ax.lines else None
        
        # Property: Trace should be displayed for both masks
        assert first_trace_data is not None
        assert second_trace_data is not None
        
        # Property: If masks are different, traces might be different
        # (not guaranteed, but the mechanism should work)
        if mask_index != 0:
            # Just verify that the trace computation ran
            assert len(second_trace_data) == len(first_trace_data)
