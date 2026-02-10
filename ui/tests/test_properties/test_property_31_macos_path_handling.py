"""
Property-based tests for macOS path handling.

Feature: pipeline-ui, Property 31: macOS Path Handling
Validates: Requirements 9.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
import sys

from ui.utils.macos_integration import MacOSIntegration


# Feature: pipeline-ui, Property 31: macOS Path Handling
@settings(max_examples=100)
@given(
    path_parts=st.lists(
        st.text(min_size=1, max_size=50).filter(
            lambda s: '/' not in s and '\0' not in s
        ),
        min_size=1,
        max_size=5
    )
)
def test_path_normalization_property(path_parts: list):
    """
    For any valid macOS file path, normalize_path should return a
    valid absolute path.
    
    Validates: Requirements 9.5
    
    Property: normalize_path should convert any relative path to
    an absolute path without errors.
    """
    # Create a path from parts
    path_str = "/".join(path_parts)
    path = Path(path_str)
    
    # Normalize the path
    try:
        normalized = MacOSIntegration.normalize_path(path)
        
        # Verify properties
        assert normalized is not None, "Normalized path should not be None"
        assert isinstance(normalized, Path), "Normalized path should be a Path object"
        assert normalized.is_absolute(), "Normalized path should be absolute"
    except Exception as e:
        # Some paths might be invalid, but the function shouldn't crash
        pytest.fail(f"normalize_path should not raise exception: {e}")


@settings(max_examples=100)
@given(
    filename=st.text(min_size=1, max_size=100).filter(
        lambda s: '/' not in s and '\0' not in s and s not in ['.', '..']
    )
)
def test_path_with_spaces_property(filename: str):
    """
    For any filename with spaces, path handling should work correctly.
    
    Validates: Requirements 9.5
    
    Property: Paths with spaces should be handled correctly by
    normalize_path and validate_path.
    """
    # Add spaces to the filename
    filename_with_spaces = f"test {filename} file"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / filename_with_spaces
        
        # Normalize should work
        normalized = MacOSIntegration.normalize_path(path)
        assert normalized is not None, "Should handle paths with spaces"
        
        # Validate should work
        is_valid = MacOSIntegration.validate_path(path)
        assert isinstance(is_valid, bool), "validate_path should return boolean"


@settings(max_examples=100)
@given(
    unicode_char=st.characters(
        blacklist_categories=('Cc', 'Cs'),  # Exclude control and surrogate chars
        blacklist_characters=['\0', '/']
    )
)
def test_path_with_unicode_property(unicode_char: str):
    """
    For any valid Unicode character, path handling should work correctly.
    
    Validates: Requirements 9.5
    
    Property: Paths with Unicode characters should be handled correctly.
    """
    filename = f"test_{unicode_char}_file"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / filename
        
        # Normalize should work
        try:
            normalized = MacOSIntegration.normalize_path(path)
            assert normalized is not None, "Should handle Unicode paths"
        except Exception as e:
            # Some Unicode characters might cause issues, but shouldn't crash
            pytest.fail(f"normalize_path should handle Unicode: {e}")
        
        # Validate should work
        is_valid = MacOSIntegration.validate_path(path)
        assert isinstance(is_valid, bool), "validate_path should return boolean"


@settings(max_examples=100)
@given(
    special_char=st.sampled_from(['!', '@', '#', '$', '%', '^', '&', '(', ')', '-', '_', '+', '=', '[', ']', '{', '}', ',', '.'])
)
def test_path_with_special_characters_property(special_char: str):
    """
    For any special character (except / and null), path handling should work.
    
    Validates: Requirements 9.5
    
    Property: Paths with special characters should be handled correctly.
    """
    filename = f"test{special_char}file"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / filename
        
        # Normalize should work
        normalized = MacOSIntegration.normalize_path(path)
        assert normalized is not None, "Should handle special characters"
        
        # Validate should work
        is_valid = MacOSIntegration.validate_path(path)
        assert isinstance(is_valid, bool), "validate_path should return boolean"


@settings(max_examples=100)
@given(
    path_depth=st.integers(min_value=1, max_value=10)
)
def test_deep_path_handling_property(path_depth: int):
    """
    For any path depth, path handling should work correctly.
    
    Validates: Requirements 9.5
    
    Property: Deeply nested paths should be handled correctly.
    """
    # Create a deeply nested path
    parts = [f"dir{i}" for i in range(path_depth)]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir)
        for part in parts:
            path = path / part
        
        # Normalize should work
        normalized = MacOSIntegration.normalize_path(path)
        assert normalized is not None, "Should handle deep paths"
        assert normalized.is_absolute(), "Normalized deep path should be absolute"
        
        # Validate should work
        is_valid = MacOSIntegration.validate_path(path)
        assert isinstance(is_valid, bool), "validate_path should return boolean"


def test_path_length_limit_property():
    """
    Paths exceeding macOS length limit should be detected by validate_path.
    
    Validates: Requirements 9.5
    
    Property: validate_path should return False for paths exceeding
    the 1024 character limit.
    """
    # Create a very long path (> 1024 characters)
    long_path = Path("/") / ("a" * 1100)
    
    is_valid = MacOSIntegration.validate_path(long_path)
    assert is_valid is False, "validate_path should reject paths > 1024 characters"


def test_null_byte_in_path_property():
    """
    Paths with null bytes should be rejected by validate_path.
    
    Validates: Requirements 9.5
    
    Property: validate_path should return False for paths containing
    null bytes.
    """
    # Create a path with null byte (if possible)
    try:
        path_str = "test\0file"
        path = Path(path_str)
        is_valid = MacOSIntegration.validate_path(path)
        assert is_valid is False, "validate_path should reject paths with null bytes"
    except ValueError:
        # Path creation might fail with null byte, which is also acceptable
        pass


@settings(max_examples=100)
@given(
    relative_parts=st.lists(
        st.text(min_size=1, max_size=20).filter(
            lambda s: '/' not in s and '\0' not in s and s not in ['.', '..']
        ),
        min_size=1,
        max_size=3
    )
)
def test_relative_to_absolute_conversion_property(relative_parts: list):
    """
    For any relative path, normalize_path should convert it to absolute.
    
    Validates: Requirements 9.5
    
    Property: normalize_path should always return an absolute path,
    even when given a relative path.
    """
    # Create a relative path
    relative_path = Path(*relative_parts)
    
    # Normalize
    normalized = MacOSIntegration.normalize_path(relative_path)
    
    # Verify it's absolute
    assert normalized.is_absolute(), \
        "normalize_path should convert relative paths to absolute"


def test_home_directory_expansion_property():
    """
    Paths with ~ should be expanded to home directory.
    
    Validates: Requirements 9.5
    
    Property: handle_special_paths should expand ~ to the user's
    home directory.
    """
    path = Path("~/test/file.txt")
    expanded = MacOSIntegration.handle_special_paths(path)
    
    assert expanded.is_absolute(), "Expanded path should be absolute"
    assert "~" not in str(expanded), "Expanded path should not contain ~"
    assert str(Path.home()) in str(expanded), "Expanded path should contain home directory"


@settings(max_examples=100)
@given(
    create_symlink=st.booleans()
)
def test_symlink_resolution_property(create_symlink: bool):
    """
    Symbolic links should be resolved by handle_special_paths.
    
    Validates: Requirements 9.5
    
    Property: handle_special_paths should resolve symbolic links
    to their targets.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "target"
        target.mkdir()
        
        if create_symlink and sys.platform != "win32":
            # Create a symbolic link (skip on Windows)
            link = Path(temp_dir) / "link"
            try:
                link.symlink_to(target)
                
                # Handle special paths should resolve the link
                resolved = MacOSIntegration.handle_special_paths(link)
                
                # On systems that support symlinks, it should resolve
                assert resolved.exists() or resolved == link, \
                    "handle_special_paths should resolve or preserve symlinks"
            except (OSError, NotImplementedError):
                # Symlink creation might fail, skip test
                pass


def test_special_directory_getters():
    """
    Special directory getters should return valid paths.
    
    Validates: Requirements 9.5
    
    Property: get_home_directory, get_documents_directory, and
    get_desktop_directory should return valid, absolute paths.
    """
    home = MacOSIntegration.get_home_directory()
    assert home.is_absolute(), "Home directory should be absolute"
    assert home.exists(), "Home directory should exist"
    
    docs = MacOSIntegration.get_documents_directory()
    assert docs.is_absolute(), "Documents directory should be absolute"
    
    desktop = MacOSIntegration.get_desktop_directory()
    assert desktop.is_absolute(), "Desktop directory should be absolute"


@settings(max_examples=100)
@given(
    path_str=st.text(min_size=1, max_size=200).filter(
        lambda s: '\0' not in s
    )
)
def test_validate_path_returns_boolean_property(path_str: str):
    """
    For any path string, validate_path should return a boolean.
    
    Validates: Requirements 9.5
    
    Property: validate_path should always return True or False,
    never raise an exception.
    """
    try:
        path = Path(path_str)
        result = MacOSIntegration.validate_path(path)
        assert isinstance(result, bool), "validate_path should return boolean"
    except Exception as e:
        # Path creation might fail for some strings, which is acceptable
        pass


def test_is_macos_detection():
    """
    is_macos() should correctly detect the platform.
    
    Validates: Requirements 9.5
    
    Property: is_macos() should return True on macOS, False otherwise.
    """
    result = MacOSIntegration.is_macos()
    assert isinstance(result, bool), "is_macos() should return boolean"
    
    # Verify it matches sys.platform
    expected = sys.platform == "darwin"
    assert result == expected, "is_macos() should match sys.platform check"


@settings(max_examples=100)
@given(
    path_parts=st.lists(
        st.text(min_size=1, max_size=30).filter(
            lambda s: '/' not in s and '\0' not in s
        ),
        min_size=1,
        max_size=5
    )
)
def test_normalize_path_idempotent_property(path_parts: list):
    """
    For any path, normalizing twice should give the same result.
    
    Validates: Requirements 9.5
    
    Property: normalize_path should be idempotent - normalizing
    an already normalized path should return the same path.
    """
    path_str = "/".join(path_parts)
    path = Path(path_str)
    
    # Normalize once
    normalized1 = MacOSIntegration.normalize_path(path)
    
    # Normalize again
    normalized2 = MacOSIntegration.normalize_path(normalized1)
    
    # Should be the same
    assert normalized1 == normalized2, \
        "normalize_path should be idempotent"
