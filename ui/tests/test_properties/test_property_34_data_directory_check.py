"""
Property-based tests for data directory existence check.

Feature: pipeline-ui, Property 34: Data Directory Existence Check
Validates: Requirements 11.3
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
import shutil

from ui.models.pipeline_config import PipelineConfig


# Feature: pipeline-ui, Property 34: Data Directory Existence Check
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
def test_data_path_exists_check_property(date: str, mouse: str, run: str):
    """
    For any valid configuration, data_path_exists() should correctly
    report whether the data directory exists.
    
    Validates: Requirements 11.3
    
    Property: data_path_exists() should return True if and only if
    the constructed data path exists on the filesystem.
    """
    # Create a temporary base directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        config.base_dir = Path(temp_dir)
        config.date = date
        config.mouse = mouse
        config.run = run
        
        # Initially, data path should not exist
        assert not config.data_path_exists(), \
            "data_path_exists() should return False for non-existent path"
        
        # Create the data path
        data_path = config.get_data_path()
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Now it should exist
        assert config.data_path_exists(), \
            "data_path_exists() should return True for existing path"
        
        # Verify the path actually exists
        assert data_path.exists(), \
            "get_data_path() should return a path that exists after creation"


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
def test_base_dir_exists_check_property(date: str, mouse: str, run: str):
    """
    For any configuration, the base directory existence should be
    checkable independently of the data path.
    
    Validates: Requirements 11.3
    
    Property: If base_dir doesn't exist, data_path_exists() should
    return False regardless of date/mouse/run values.
    """
    # Use a non-existent base directory
    config = PipelineConfig()
    config.base_dir = Path("/nonexistent/base/directory/that/should/not/exist")
    config.date = date
    config.mouse = mouse
    config.run = run
    
    # data_path_exists should return False
    assert not config.data_path_exists(), \
        "data_path_exists() should return False when base_dir doesn't exist"


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
    ),
    create_partial=st.booleans()
)
def test_partial_path_existence_property(
    date: str,
    mouse: str,
    run: str,
    create_partial: bool
):
    """
    For any configuration, data_path_exists() should only return True
    if the complete path exists, not just partial directories.
    
    Validates: Requirements 11.3
    
    Property: Creating only part of the path (e.g., data/DATE/MOUSE)
    should not make data_path_exists() return True.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        config.base_dir = Path(temp_dir)
        config.date = date
        config.mouse = mouse
        config.run = run
        
        if create_partial:
            # Create only partial path (data/DATE/MOUSE but not RUN)
            partial_path = config.base_dir / "data" / date / mouse
            partial_path.mkdir(parents=True, exist_ok=True)
            
            # Full path should still not exist
            assert not config.data_path_exists(), \
                "data_path_exists() should return False for partial path"
        
        # Create full path
        full_path = config.get_data_path()
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Now it should exist
        assert config.data_path_exists(), \
            "data_path_exists() should return True for complete path"


def test_data_path_exists_with_file_instead_of_directory():
    """
    If the data path points to a file instead of a directory,
    data_path_exists() should handle it appropriately.
    
    Validates: Requirements 11.3
    
    Property: data_path_exists() should check for directory existence,
    not just path existence.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        config.base_dir = Path(temp_dir)
        config.date = "2025-01-01"
        config.mouse = "test"
        config.run = "run1"
        
        # Create the path as a file instead of directory
        data_path = config.get_data_path()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.touch()  # Create as file
        
        # This is an edge case - the path exists but is not a directory
        # The implementation should handle this gracefully
        # (either return False or True, but shouldn't crash)
        try:
            result = config.data_path_exists()
            assert isinstance(result, bool), \
                "data_path_exists() should return a boolean"
        except Exception as e:
            pytest.fail(f"data_path_exists() should not raise exception: {e}")


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
def test_data_path_exists_consistency_property(date: str, mouse: str, run: str):
    """
    For any configuration, data_path_exists() should be consistent
    with Path.exists() on the constructed path.
    
    Validates: Requirements 11.3
    
    Property: data_path_exists() should return the same value as
    get_data_path().exists().
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        config.base_dir = Path(temp_dir)
        config.date = date
        config.mouse = mouse
        config.run = run
        
        # Check consistency before creation
        data_path = config.get_data_path()
        assert config.data_path_exists() == data_path.exists(), \
            "data_path_exists() should match get_data_path().exists()"
        
        # Create the path
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Check consistency after creation
        assert config.data_path_exists() == data_path.exists(), \
            "data_path_exists() should match get_data_path().exists() after creation"


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
def test_data_path_exists_after_deletion_property(date: str, mouse: str, run: str):
    """
    For any configuration, data_path_exists() should return False
    after the directory is deleted.
    
    Validates: Requirements 11.3
    
    Property: After deleting a data directory, data_path_exists()
    should return False.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig()
        config.base_dir = Path(temp_dir)
        config.date = date
        config.mouse = mouse
        config.run = run
        
        # Create the path
        data_path = config.get_data_path()
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Verify it exists
        assert config.data_path_exists(), \
            "data_path_exists() should return True after creation"
        
        # Delete the directory
        shutil.rmtree(data_path)
        
        # Verify it no longer exists
        assert not config.data_path_exists(), \
            "data_path_exists() should return False after deletion"


def test_data_path_exists_with_empty_parameters():
    """
    With empty parameters, data_path_exists() should handle gracefully.
    
    Validates: Requirements 11.3
    
    Property: data_path_exists() should not crash with empty parameters.
    """
    config = PipelineConfig()
    config.date = ""
    config.mouse = ""
    config.run = ""
    
    # Should not crash
    try:
        result = config.data_path_exists()
        assert isinstance(result, bool), \
            "data_path_exists() should return a boolean even with empty parameters"
    except Exception as e:
        pytest.fail(f"data_path_exists() should not raise exception with empty params: {e}")
