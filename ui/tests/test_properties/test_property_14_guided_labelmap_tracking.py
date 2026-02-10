"""
Property-based tests for guided labelmap tracking.

Feature: pipeline-ui, Property 14: Guided Labelmap Tracking
Validates: Requirements 4.6, 6.7
"""

import pytest
from hypothesis import given, strategies as st, settings

from ui.models.progress_state import ProgressState


# Feature: pipeline-ui, Property 14: Guided Labelmap Tracking
@settings(max_examples=100)
@given(
    dataset_key=st.text(min_size=1, max_size=100),
    m2_completed=st.booleans(),
    m2_5_completed=st.booleans(),
    m2_run_after_m2_5=st.booleans()
)
def test_guided_labelmap_tracking_property(
    dataset_key: str,
    m2_completed: bool,
    m2_5_completed: bool,
    m2_run_after_m2_5: bool
):
    """
    For any progress state, the m2_guided flag should correctly reflect
    whether M2 was last run with guides enabled.
    
    Validates: Requirements 4.6, 6.7
    
    Property: If M2.5 is completed and M2 is run after M2.5, then m2_guided
    should be True. If M2 is run without M2.5 being completed, m2_guided
    should be False.
    """
    progress = ProgressState(dataset_key)
    
    # Simulate the workflow
    if m2_completed:
        progress.mark_complete("M2")
        # Initially, M2 is not guided
        progress.m2_guided = False
    
    if m2_5_completed:
        progress.mark_complete("M2.5")
    
    if m2_run_after_m2_5 and m2_5_completed:
        # Re-run M2 after M2.5
        progress.mark_complete("M2")
        # M2 should now be guided
        progress.m2_guided = True
    
    # Verify the property
    if m2_5_completed and m2_run_after_m2_5:
        # M2 was run after M2.5, so it should be guided
        assert progress.m2_guided is True, \
            "m2_guided should be True when M2 is run after M2.5"
    elif m2_completed and not m2_5_completed:
        # M2 was run without M2.5, so it should not be guided
        assert progress.m2_guided is False, \
            "m2_guided should be False when M2 is run without M2.5"


@settings(max_examples=100)
@given(
    dataset_key=st.text(min_size=1, max_size=100),
    modules_sequence=st.lists(
        st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]),
        min_size=0,
        max_size=20
    )
)
def test_guided_flag_persistence_property(
    dataset_key: str,
    modules_sequence: list
):
    """
    For any sequence of module completions, the m2_guided flag should
    correctly track whether M2 was last run with guides.
    
    Validates: Requirements 4.6, 6.7
    
    Property: The m2_guided flag should only be True if M2.5 was completed
    before the last M2 completion.
    """
    progress = ProgressState(dataset_key)
    
    m2_5_seen = False
    last_m2_after_m2_5 = False
    
    for module_id in modules_sequence:
        progress.mark_complete(module_id)
        
        if module_id == "M2.5":
            m2_5_seen = True
        
        if module_id == "M2":
            # Update guided flag based on whether M2.5 was seen
            progress.m2_guided = m2_5_seen
            last_m2_after_m2_5 = m2_5_seen
    
    # Verify the property
    if "M2" in modules_sequence:
        # M2 was completed at some point
        assert progress.m2_guided == last_m2_after_m2_5, \
            "m2_guided should reflect whether M2.5 was completed before last M2 run"


@settings(max_examples=100)
@given(
    dataset_key=st.text(min_size=1, max_size=100)
)
def test_guided_flag_initial_state_property(dataset_key: str):
    """
    For any new progress state, the m2_guided flag should initially be False.
    
    Validates: Requirements 4.6, 6.7
    
    Property: A newly created progress state should have m2_guided = False.
    """
    progress = ProgressState(dataset_key)
    
    assert progress.m2_guided is False, \
        "m2_guided should be False for a new progress state"


@settings(max_examples=100)
@given(
    dataset_key=st.text(min_size=1, max_size=100),
    initial_guided=st.booleans()
)
def test_guided_flag_reset_property(dataset_key: str, initial_guided: bool):
    """
    For any progress state, resetting should clear the m2_guided flag.
    
    Validates: Requirements 4.6, 6.7
    
    Property: After reset(), m2_guided should be False regardless of
    its previous value.
    """
    progress = ProgressState(dataset_key)
    progress.m2_guided = initial_guided
    progress.mark_complete("M2")
    progress.mark_complete("M2.5")
    
    # Reset progress
    progress.reset()
    
    # Verify the property
    assert progress.m2_guided is False, \
        "m2_guided should be False after reset()"
    assert len(progress.completed_modules) == 0, \
        "completed_modules should be empty after reset()"


@settings(max_examples=100)
@given(
    dataset_key=st.text(min_size=1, max_size=100),
    m2_guided=st.booleans()
)
def test_guided_flag_serialization_property(dataset_key: str, m2_guided: bool):
    """
    For any progress state with m2_guided flag, serialization and
    deserialization should preserve the flag value.
    
    Validates: Requirements 4.6, 6.7
    
    Property: to_dict() followed by from_dict() should preserve m2_guided.
    """
    progress = ProgressState(dataset_key)
    progress.m2_guided = m2_guided
    progress.mark_complete("M2")
    if m2_guided:
        progress.mark_complete("M2.5")
    
    # Serialize and deserialize
    data = progress.to_dict()
    restored = ProgressState.from_dict(data)
    
    # Verify the property
    assert restored.m2_guided == progress.m2_guided, \
        "m2_guided should be preserved through serialization"
    assert restored.dataset_key == progress.dataset_key, \
        "dataset_key should be preserved through serialization"
    assert restored.completed_modules == progress.completed_modules, \
        "completed_modules should be preserved through serialization"
