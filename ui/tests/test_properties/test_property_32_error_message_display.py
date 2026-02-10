"""
Property-based tests for error message display.

Feature: pipeline-ui, Property 32: Error Message Display
Validates: Requirements 10.3
"""

import pytest
from hypothesis import given, strategies as st, settings
from pathlib import Path

from ui.utils.error_messages import ErrorMessages, WarningMessages


# Feature: pipeline-ui, Property 32: Error Message Display
@settings(max_examples=100)
@given(
    path_str=st.text(min_size=1, max_size=200)
)
def test_error_message_exists_for_missing_data_dir(path_str: str):
    """
    For any path, there should be an appropriate error message for
    missing data directory.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.format_missing_data_dir should return a
    non-empty, user-friendly message for any path.
    """
    path = Path(path_str)
    message = ErrorMessages.format_missing_data_dir(path)
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert str(path) in message, "Error message should contain the path"
    assert "data directory" in message.lower() or "directory" in message.lower(), \
        "Error message should mention directory"


@settings(max_examples=100)
@given(
    exit_code=st.integers(min_value=1, max_value=255)
)
def test_error_message_exists_for_script_failure(exit_code: int):
    """
    For any script exit code, there should be an appropriate error message.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.format_script_failed should return a
    non-empty, user-friendly message for any exit code.
    """
    message = ErrorMessages.format_script_failed(exit_code)
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert str(exit_code) in message, "Error message should contain the exit code"
    assert "failed" in message.lower() or "error" in message.lower(), \
        "Error message should indicate failure"


@settings(max_examples=100)
@given(
    files=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10)
)
def test_error_message_exists_for_missing_inputs(files: list):
    """
    For any list of missing files, there should be an appropriate error message.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.format_missing_inputs should return a
    non-empty, user-friendly message for any list of files.
    """
    message = ErrorMessages.format_missing_inputs(files)
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert "missing" in message.lower() or "required" in message.lower(), \
        "Error message should indicate missing files"
    
    # Check that at least some files are mentioned
    for file in files[:3]:  # Check first few files
        assert file in message, f"Error message should contain file: {file}"


@settings(max_examples=100)
@given(
    files=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10)
)
def test_error_message_exists_for_missing_outputs(files: list):
    """
    For any list of missing outputs, there should be an appropriate error message.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.format_missing_outputs should return a
    non-empty, user-friendly message for any list of files.
    """
    message = ErrorMessages.format_missing_outputs(files)
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert "output" in message.lower() or "missing" in message.lower(), \
        "Error message should indicate missing outputs"


@settings(max_examples=100)
@given(
    timeout=st.integers(min_value=1, max_value=3600)
)
def test_error_message_exists_for_timeout(timeout: int):
    """
    For any timeout value, there should be an appropriate error message.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.format_script_timeout should return a
    non-empty, user-friendly message for any timeout value.
    """
    message = ErrorMessages.format_script_timeout(timeout)
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert str(timeout) in message, "Error message should contain the timeout value"
    assert "timeout" in message.lower() or "timed out" in message.lower(), \
        "Error message should indicate timeout"


def test_error_message_exists_for_invalid_date():
    """
    There should be an appropriate error message for invalid date format.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.INVALID_DATE should be a non-empty,
    user-friendly message.
    """
    message = ErrorMessages.INVALID_DATE
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert "date" in message.lower(), "Error message should mention date"
    assert "YYYY-MM-DD" in message, "Error message should show expected format"


def test_error_message_exists_for_invalid_identifier():
    """
    There should be an appropriate error message for invalid identifier.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.INVALID_IDENTIFIER should be a non-empty,
    user-friendly message.
    """
    message = ErrorMessages.INVALID_IDENTIFIER
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert "alphanumeric" in message.lower() or "letters" in message.lower(), \
        "Error message should describe valid characters"


def test_error_message_exists_for_missing_venv():
    """
    There should be an appropriate error message for missing virtual environment.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.MISSING_VENV should be a non-empty,
    user-friendly message with suggestions.
    """
    message = ErrorMessages.MISSING_VENV
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert "virtual environment" in message.lower() or "venv" in message.lower(), \
        "Error message should mention virtual environment"
    assert ".venv" in message or "venv" in message, \
        "Error message should suggest venv directory names"


def test_error_message_exists_for_execution_in_progress():
    """
    There should be an appropriate error message for execution already in progress.
    
    Validates: Requirements 10.3
    
    Property: ErrorMessages.EXECUTION_IN_PROGRESS should be a non-empty,
    user-friendly message.
    """
    message = ErrorMessages.EXECUTION_IN_PROGRESS
    
    assert message is not None, "Error message should not be None"
    assert len(message) > 0, "Error message should not be empty"
    assert "running" in message.lower() or "progress" in message.lower(), \
        "Error message should indicate execution in progress"


@settings(max_examples=100)
@given(
    path_str=st.text(min_size=1, max_size=200)
)
def test_warning_message_exists_for_data_dir_not_exist(path_str: str):
    """
    For any path, there should be an appropriate warning message for
    non-existent data directory.
    
    Validates: Requirements 10.3
    
    Property: WarningMessages.format_data_dir_not_exist should return a
    non-empty, user-friendly warning for any path.
    """
    path = Path(path_str)
    message = WarningMessages.format_data_dir_not_exist(path)
    
    assert message is not None, "Warning message should not be None"
    assert len(message) > 0, "Warning message should not be empty"
    assert str(path) in message, "Warning message should contain the path"


@settings(max_examples=100)
@given(
    description=st.text(min_size=1, max_size=200)
)
def test_warning_message_exists_for_optional_module(description: str):
    """
    For any module description, there should be an appropriate warning
    message for optional modules.
    
    Validates: Requirements 10.3
    
    Property: WarningMessages.format_optional_module should return a
    non-empty, user-friendly warning for any description.
    """
    message = WarningMessages.format_optional_module(description)
    
    assert message is not None, "Warning message should not be None"
    assert len(message) > 0, "Warning message should not be empty"
    assert "optional" in message.lower(), "Warning message should mention optional"
    assert description in message, "Warning message should contain the description"


def test_all_error_messages_are_user_friendly():
    """
    All error messages should be user-friendly (no technical jargon).
    
    Validates: Requirements 10.3
    
    Property: Error messages should avoid technical terms and provide
    clear guidance.
    """
    # Check that messages provide suggestions or guidance
    messages_with_suggestions = [
        ErrorMessages.MISSING_DATA_DIR,
        ErrorMessages.MISSING_VENV,
        ErrorMessages.SCRIPT_FAILED,
        ErrorMessages.MISSING_INPUTS,
        ErrorMessages.MISSING_OUTPUTS,
    ]
    
    for message in messages_with_suggestions:
        assert "suggestion" in message.lower() or "please" in message.lower() or \
               "try" in message.lower() or "check" in message.lower(), \
            f"Error message should provide guidance: {message[:50]}..."


def test_error_messages_are_not_empty():
    """
    All predefined error messages should be non-empty.
    
    Validates: Requirements 10.3
    
    Property: No error message constant should be empty.
    """
    error_attrs = [
        attr for attr in dir(ErrorMessages)
        if not attr.startswith('_') and not callable(getattr(ErrorMessages, attr))
    ]
    
    for attr in error_attrs:
        if attr.startswith('format_'):
            continue  # Skip format methods
        message = getattr(ErrorMessages, attr)
        if isinstance(message, str):
            assert len(message) > 0, f"Error message {attr} should not be empty"
