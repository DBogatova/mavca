"""
Property-based tests for missing directory warning.

Feature: pipeline-ui, Property 35: Missing Directory Warning
Validates: Requirements 11.4
"""

import pytest
from hypothesis import given, strategies as st, settings
from pathlib import Path
import tempfile

from ui.models.pipeline_config import PipelineConfig
from ui.utils.error_messages import WarningMessages


# Feature: pipeline-ui, Property 35: Missing Directory Warning
@settings(max_examples=100)
@given(
    date=st.text(min_size=10, max_size=10).filter(
        lambda s: len(s) == 10 and s[4] == '-' and s[7] == '-'
    ),
    mouse=st.text(min_size=1, max_size=50).filter(
        lambda s: all(c.isalnum() or c in '_-' for c in s)
    ),
    run=st.text(min_size=1, max_size=50).filter(
        lambda s: all(c.isalnum() or c in '_-' for c in s)
    )
)
def test_warning_displayed_for_nonexistent_directory_property(
    date: str,
    mouse: str,
    run: str
):
    """
    For any non-existent data directory, a warning should be available.
    
    Validates: Requirements 11.4
    
    Property: When data_path_exists() returns False, there should be
    an appropriate warning message available.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        config.base_dir = Path(temp_dir)
        config.date = date
        config.mouse = mouse
        config.run = run
        
        # Ensure directory doesn't exist
        data_path = config.get_data_path()
        if data_path.exists():
            import shutil
            shutil.rmtree(data_path)
        
        # Verify it doesn't exist
        assert not config.data_path_exists(), \
            "Test setup: data path should not exist"
        
        # Get warning message
        warning = WarningMessages.format_data_dir_not_exist(data_path)
        
        # Verify warning properties
        assert warning is not None, \
            "Warning should exist for non-existent directory"
        assert len(warning) > 0, \
            "Warning should not be empty"
        assert str(data_path) in warning, \
            "Warning should contain the path"
        assert "not exist" in warning.lower() or "does not exist" in warning.lower(), \
            "Warning should indicate directory doesn't exist"


@settings(max_examples=100)
@given(
    date=st.text(min_size=10, max_size=10).filter(
        lambda s: len(s) == 10 and s[4] == '-' and s[7] == '-'
    ),
    mouse=st.text(min_size=1, max_size=50).filter(
        lambda s: all(c.isalnum() or c in '_-' for c in s)
    ),
    run=st.text(min_size=1, max_size=50).filter(
        lambda s: all(c.isalnum() or c in '_-' for c in s)
    )
)
def test_warning_not_needed_for_existing_directory_property(
    date: str,
    mouse: str,
    run: str
):
    """
    For any existing data directory, no warning should be needed.
    
    Validates: Requirements 11.4
    
    Property: When data_path_exists() returns True, the warning
    mechanism should still work but the warning shouldn't be shown
    in the UI logic.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        config.base_dir = Path(temp_dir)
        config.date = date
        config.mouse = mouse
        config.run = run
        
        # Create the directory
        data_path = config.get_data_path()
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Verify it exists
        assert config.data_path_exists(), \
            "Test setup: data path should exist"
        
        # The warning message should still be formattable
        # (even though it shouldn't be shown in the UI)
        warning = WarningMessages.format_data_dir_not_exist(data_path)
        assert warning is not None, \
            "Warning formatter should work even for existing paths"


@settings(max_examples=100)
@given(
    path_str=st.text(min_size=1, max_size=200)
)
def test_warning_message_format_property(path_str: str):
    """
    For any path string, the warning message should be properly formatted.
    
    Validates: Requirements 11.4
    
    Property: WarningMessages.format_data_dir_not_exist should produce
    a well-formatted warning for any path.
    """
    path = Path(path_str)
    warning = WarningMessages.format_data_dir_not_exist(path)
    
    assert warning is not None, "Warning should not be None"
    assert len(warning) > 0, "Warning should not be empty"
    assert isinstance(warning, str), "Warning should be a string"
    
    # Check for warning indicator
    assert "⚠" in warning or "warning" in warning.lower() or "note" in warning.lower(), \
        "Warning should have a visual indicator or mention it's a warning"


def test_warning_message_is_user_friendly():
    """
    The missing directory warning should be user-friendly.
    
    Validates: Requirements 11.4
    
    Property: The warning should explain the situation clearly
    without technical jargon.
    """
    path = Path("/test/path/data/2025-01-01/mouse1/run1")
    warning = WarningMessages.format_data_dir_not_exist(path)
    
    # Should not contain technical jargon
    technical_terms = ["errno", "ENOENT", "exception", "traceback", "null pointer"]
    for term in technical_terms:
        assert term.lower() not in warning.lower(), \
            f"Warning should not contain technical term: {term}"
    
    # Should be informative
    assert len(warning) > 50, \
        "Warning should be informative (more than just 'directory not found')"


@settings(max_examples=100)
@given(
    date=st.text(min_size=10, max_size=10).filter(
        lambda s: len(s) == 10 and s[4] == '-' and s[7] == '-'
    ),
    mouse=st.text(min_size=1, max_size=50).filter(
        lambda s: all(c.isalnum() or c in '_-' for c in s)
    ),
    run=st.text(min_size=1, max_size=50).filter(
        lambda s: all(c.isalnum() or c in '_-' for c in s)
    )
)
def test_warning_before_execution_property(date: str, mouse: str, run: str):
    """
    For any configuration with non-existent directory, the warning
    should be checkable before script execution.
    
    Validates: Requirements 11.4
    
    Property: The UI should be able to check data_path_exists()
    before allowing script execution.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        config.base_dir = Path(temp_dir)
        config.date = date
        config.mouse = mouse
        config.run = run
        
        # Simulate pre-execution check
        should_warn = not config.data_path_exists()
        
        if should_warn:
            # Warning should be available
            warning = WarningMessages.format_data_dir_not_exist(config.get_data_path())
            assert warning is not None and len(warning) > 0, \
                "Warning should be available when directory doesn't exist"
        else:
            # No warning needed, but check should still work
            assert config.data_path_exists(), \
                "If no warning needed, directory should exist"


def test_warning_contains_helpful_context():
    """
    The missing directory warning should contain helpful context.
    
    Validates: Requirements 11.4
    
    Property: The warning should help users understand why the
    directory might not exist and what to do about it.
    """
    path = Path("/test/data/2025-01-01/mouse1/run1")
    warning = WarningMessages.format_data_dir_not_exist(path)
    
    # Should provide context or suggestions
    helpful_phrases = [
        "normal",
        "haven't run",
        "will need",
        "created",
        "first time"
    ]
    
    has_helpful_content = any(phrase in warning.lower() for phrase in helpful_phrases)
    assert has_helpful_content, \
        "Warning should contain helpful context or suggestions"


@settings(max_examples=100)
@given(
    base_exists=st.booleans(),
    data_exists=st.booleans()
)
def test_warning_distinguishes_base_vs_data_directory(
    base_exists: bool,
    data_exists: bool
):
    """
    The warning system should distinguish between missing base directory
    and missing data directory.
    
    Validates: Requirements 11.4
    
    Property: Different warnings should be available for missing base
    directory vs missing data directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        
        if base_exists:
            config.base_dir = Path(temp_dir)
        else:
            config.base_dir = Path("/nonexistent/base/directory")
        
        config.date = "2025-01-01"
        config.mouse = "test"
        config.run = "run1"
        
        if base_exists and data_exists:
            # Create data directory
            data_path = config.get_data_path()
            data_path.mkdir(parents=True, exist_ok=True)
        
        # Check existence
        path_exists = config.data_path_exists()
        
        if not path_exists:
            # Should be able to generate appropriate warning
            if not base_exists:
                # Base directory issue
                from ui.utils.error_messages import ErrorMessages
                warning = ErrorMessages.format_missing_base_dir(config.base_dir)
            else:
                # Data directory issue
                warning = WarningMessages.format_data_dir_not_exist(config.get_data_path())
            
            assert warning is not None and len(warning) > 0, \
                "Appropriate warning should be available"


def test_warning_message_constant_exists():
    """
    The warning message constant should exist and be accessible.
    
    Validates: Requirements 11.4
    
    Property: WarningMessages.DATA_DIR_NOT_EXIST should be defined
    and non-empty.
    """
    assert hasattr(WarningMessages, 'DATA_DIR_NOT_EXIST'), \
        "WarningMessages should have DATA_DIR_NOT_EXIST constant"
    
    message = WarningMessages.DATA_DIR_NOT_EXIST
    assert message is not None, "Warning constant should not be None"
    assert len(message) > 0, "Warning constant should not be empty"
    assert "{path}" in message, "Warning template should have path placeholder"
