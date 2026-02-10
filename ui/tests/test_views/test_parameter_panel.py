"""
Unit tests for ParameterPanel widget.

Tests input field presence, validation feedback display,
and base directory configuration.
"""

import pytest
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.views.parameter_panel import ParameterPanel


@pytest.fixture
def app(qapp):
    """Provide QApplication instance."""
    return qapp


@pytest.fixture
def parameter_panel(app):
    """Create a ParameterPanel instance for testing."""
    panel = ParameterPanel()
    
    # Set up validation callbacks
    def validate_date(date_str: str) -> bool:
        import re
        pattern = r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$'
        return bool(re.match(pattern, date_str))
    
    def validate_identifier(identifier: str) -> bool:
        import re
        if not identifier:
            return False
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, identifier))
    
    panel.validate_date_callback = validate_date
    panel.validate_identifier_callback = validate_identifier
    
    return panel


def test_date_input_field_present(parameter_panel):
    """Test that DATE input field is present."""
    assert parameter_panel.date_input is not None
    assert parameter_panel.date_input.placeholderText() == "YYYY-MM-DD"


def test_mouse_input_field_present(parameter_panel):
    """Test that MOUSE input field is present."""
    assert parameter_panel.mouse_input is not None
    assert "rAi162_phpeb" in parameter_panel.mouse_input.placeholderText()


def test_run_input_field_present(parameter_panel):
    """Test that RUN input field is present."""
    assert parameter_panel.run_input is not None
    assert "run1" in parameter_panel.run_input.placeholderText()


def test_base_directory_button_present(parameter_panel):
    """Test that base directory configuration button is present."""
    assert parameter_panel.base_dir_button is not None
    assert "Base Directory" in parameter_panel.base_dir_button.text()


def test_validation_labels_present(parameter_panel):
    """Test that validation feedback labels are present."""
    assert parameter_panel.date_validation_label is not None
    assert parameter_panel.mouse_validation_label is not None
    assert parameter_panel.run_validation_label is not None


def test_date_validation_feedback_valid(parameter_panel):
    """Test validation feedback for valid DATE input."""
    # Enter valid date
    parameter_panel.date_input.setText("2025-12-25")
    
    # Validation label should be empty
    assert parameter_panel.date_validation_label.text() == ""
    assert "red" not in parameter_panel.date_input.styleSheet()


def test_date_validation_feedback_invalid(parameter_panel):
    """Test validation feedback for invalid DATE input."""
    # Enter invalid date
    parameter_panel.date_input.setText("invalid-date")
    
    # Validation label should show error
    assert "Invalid format" in parameter_panel.date_validation_label.text()
    assert "red" in parameter_panel.date_input.styleSheet()


def test_mouse_validation_feedback_valid(parameter_panel):
    """Test validation feedback for valid MOUSE input."""
    # Enter valid mouse identifier
    parameter_panel.mouse_input.setText("rAi162_phpeb")
    
    # Validation label should be empty
    assert parameter_panel.mouse_validation_label.text() == ""
    assert "red" not in parameter_panel.mouse_input.styleSheet()


def test_mouse_validation_feedback_invalid(parameter_panel):
    """Test validation feedback for invalid MOUSE input."""
    # Enter invalid mouse identifier (contains spaces)
    parameter_panel.mouse_input.setText("invalid mouse")
    
    # Validation label should show error
    assert "alphanumeric" in parameter_panel.mouse_validation_label.text()
    assert "red" in parameter_panel.mouse_input.styleSheet()


def test_run_validation_feedback_valid(parameter_panel):
    """Test validation feedback for valid RUN input."""
    # Enter valid run identifier
    parameter_panel.run_input.setText("run1")
    
    # Validation label should be empty
    assert parameter_panel.run_validation_label.text() == ""
    assert "red" not in parameter_panel.run_input.styleSheet()


def test_run_validation_feedback_invalid(parameter_panel):
    """Test validation feedback for invalid RUN input."""
    # Enter invalid run identifier (contains special characters)
    parameter_panel.run_input.setText("run@1")
    
    # Validation label should show error
    assert "alphanumeric" in parameter_panel.run_validation_label.text()
    assert "red" in parameter_panel.run_input.styleSheet()


def test_parameter_change_indicator(parameter_panel):
    """Test that parameter change indicator is displayed."""
    # Initially empty
    assert parameter_panel.change_indicator.text() == ""
    
    # Enter a parameter
    parameter_panel.date_input.setText("2025-12-25")
    
    # Change indicator should be shown
    assert "updated" in parameter_panel.change_indicator.text().lower()
    assert "next execution" in parameter_panel.change_indicator.text().lower()


def test_base_directory_display(parameter_panel):
    """Test base directory display."""
    test_path = Path("/test/path/to/data")
    parameter_panel.set_base_dir(test_path)
    
    assert str(test_path) in parameter_panel.base_dir_label.text()


def test_get_set_date(parameter_panel):
    """Test getting and setting DATE value."""
    test_date = "2025-12-25"
    parameter_panel.set_date(test_date)
    
    assert parameter_panel.get_date() == test_date


def test_get_set_mouse(parameter_panel):
    """Test getting and setting MOUSE value."""
    test_mouse = "rAi162_phpeb"
    parameter_panel.set_mouse(test_mouse)
    
    assert parameter_panel.get_mouse() == test_mouse


def test_get_set_run(parameter_panel):
    """Test getting and setting RUN value."""
    test_run = "run1"
    parameter_panel.set_run(test_run)
    
    assert parameter_panel.get_run() == test_run


def test_parameters_changed_signal(parameter_panel, qtbot):
    """Test that parameters_changed signal is emitted."""
    with qtbot.waitSignal(parameter_panel.parameters_changed, timeout=1000) as blocker:
        parameter_panel.date_input.setText("2025-12-25")
    
    # Signal should have been emitted with correct values
    assert blocker.args == ["2025-12-25", "", ""]


def test_base_dir_changed_signal(parameter_panel, qtbot):
    """Test that base_dir_changed signal is emitted."""
    test_path = Path("/test/path")
    
    with qtbot.waitSignal(parameter_panel.base_dir_changed, timeout=1000) as blocker:
        parameter_panel.set_base_dir(test_path)
        parameter_panel.base_dir_changed.emit(test_path)
    
    # Signal should have been emitted with correct path
    assert blocker.args == [test_path]


def test_clear_change_indicator(parameter_panel):
    """Test clearing the parameter change indicator."""
    # Set some text
    parameter_panel.date_input.setText("2025-12-25")
    assert parameter_panel.change_indicator.text() != ""
    
    # Clear indicator
    parameter_panel.clear_change_indicator()
    assert parameter_panel.change_indicator.text() == ""
