"""
Unit tests for behavioral analysis extension.

Tests the behavioral data integration placeholder and extension points.
"""

import pytest
from PyQt5.QtWidgets import QApplication

from ui.extensions.behavioral_analysis import (
    BehavioralAnalysisWidget,
    BehavioralAnalysisExtension,
    BehavioralDataType,
    get_behavioral_extension,
    show_behavioral_analysis_info
)
from pathlib import Path


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_behavioral_analysis_widget_creation(qapp):
    """Test behavioral analysis widget can be created."""
    widget = BehavioralAnalysisWidget()
    assert widget is not None


def test_behavioral_analysis_widget_has_ui_elements(qapp):
    """Test behavioral analysis widget has expected UI elements."""
    widget = BehavioralAnalysisWidget()
    
    # Should have checkboxes for data types
    assert widget.pupil_checkbox is not None
    assert widget.whisking_checkbox is not None
    assert widget.accel_checkbox is not None
    
    # Should have buttons
    assert widget.load_button is not None
    assert widget.sync_button is not None
    assert widget.analyze_button is not None
    
    # Should have status label
    assert widget.status_label is not None


def test_behavioral_analysis_widget_checkboxes_disabled(qapp):
    """Test behavioral analysis widget checkboxes are disabled (placeholder)."""
    widget = BehavioralAnalysisWidget()
    
    # Placeholder checkboxes should be disabled
    assert not widget.pupil_checkbox.isEnabled()
    assert not widget.whisking_checkbox.isEnabled()
    assert not widget.accel_checkbox.isEnabled()


def test_behavioral_analysis_widget_buttons_disabled(qapp):
    """Test behavioral analysis widget buttons are disabled (placeholder)."""
    widget = BehavioralAnalysisWidget()
    
    # Placeholder buttons should be disabled
    assert not widget.load_button.isEnabled()
    assert not widget.sync_button.isEnabled()
    assert not widget.analyze_button.isEnabled()


def test_behavioral_analysis_widget_set_data_path(qapp):
    """Test setting data path on behavioral analysis widget."""
    widget = BehavioralAnalysisWidget()
    test_path = Path("/test/data/path")
    
    widget.set_data_path(test_path)
    assert widget._data_path == test_path
    assert str(test_path) in widget.status_label.text()


def test_behavioral_analysis_widget_enable_data_type(qapp):
    """Test enabling behavioral data types."""
    widget = BehavioralAnalysisWidget()
    
    widget.enable_data_type(BehavioralDataType.PUPIL)
    assert BehavioralDataType.PUPIL in widget._enabled_data_types
    
    widget.enable_data_type(BehavioralDataType.WHISKING)
    assert BehavioralDataType.WHISKING in widget._enabled_data_types
    
    assert len(widget._enabled_data_types) == 2


def test_behavioral_analysis_widget_disable_data_type(qapp):
    """Test disabling behavioral data types."""
    widget = BehavioralAnalysisWidget()
    
    widget.enable_data_type(BehavioralDataType.PUPIL)
    widget.enable_data_type(BehavioralDataType.WHISKING)
    
    widget.disable_data_type(BehavioralDataType.PUPIL)
    assert BehavioralDataType.PUPIL not in widget._enabled_data_types
    assert BehavioralDataType.WHISKING in widget._enabled_data_types


def test_behavioral_analysis_widget_load_data_placeholder(qapp):
    """Test load_behavioral_data returns False (placeholder)."""
    widget = BehavioralAnalysisWidget()
    result = widget.load_behavioral_data(BehavioralDataType.PUPIL)
    assert result is False, "Placeholder should return False"


def test_behavioral_analysis_widget_synchronize_placeholder(qapp):
    """Test synchronize_with_imaging returns False (placeholder)."""
    widget = BehavioralAnalysisWidget()
    result = widget.synchronize_with_imaging()
    assert result is False, "Placeholder should return False"


def test_behavioral_analysis_widget_correlation_placeholder(qapp):
    """Test run_correlation_analysis returns empty dict (placeholder)."""
    widget = BehavioralAnalysisWidget()
    result = widget.run_correlation_analysis()
    assert result == {}, "Placeholder should return empty dict"


def test_behavioral_data_type_enum():
    """Test BehavioralDataType enum has expected values."""
    assert BehavioralDataType.PUPIL.value == "pupil"
    assert BehavioralDataType.WHISKING.value == "whisking"
    assert BehavioralDataType.ACCELEROMETER.value == "accelerometer"
    assert BehavioralDataType.CUSTOM.value == "custom"


def test_behavioral_analysis_extension_creation():
    """Test behavioral analysis extension can be created."""
    extension = BehavioralAnalysisExtension()
    assert extension is not None


def test_behavioral_analysis_extension_register_processor():
    """Test registering a processor with behavioral extension."""
    extension = BehavioralAnalysisExtension()
    
    # Create a mock processor
    mock_processor = {"name": "test_processor"}
    
    extension.register_processor("test", mock_processor)
    assert "test" in extension.list_processors()


def test_behavioral_analysis_extension_activate_processor():
    """Test activating a processor."""
    extension = BehavioralAnalysisExtension()
    
    mock_processor = {"name": "test_processor"}
    extension.register_processor("test", mock_processor)
    
    result = extension.activate_processor("test")
    assert result is True
    assert "test" in extension.list_active_processors()


def test_behavioral_analysis_extension_deactivate_processor():
    """Test deactivating a processor."""
    extension = BehavioralAnalysisExtension()
    
    mock_processor = {"name": "test_processor"}
    extension.register_processor("test", mock_processor)
    extension.activate_processor("test")
    
    result = extension.deactivate_processor("test")
    assert result is True
    assert "test" not in extension.list_active_processors()


def test_behavioral_analysis_extension_activate_nonexistent_processor():
    """Test activating nonexistent processor returns False."""
    extension = BehavioralAnalysisExtension()
    
    result = extension.activate_processor("nonexistent")
    assert result is False


def test_behavioral_analysis_extension_get_processor():
    """Test getting a processor."""
    extension = BehavioralAnalysisExtension()
    
    mock_processor = {"name": "test_processor"}
    extension.register_processor("test", mock_processor)
    
    processor = extension.get_processor("test")
    assert processor == mock_processor


def test_behavioral_analysis_extension_list_processors():
    """Test listing registered processors."""
    extension = BehavioralAnalysisExtension()
    
    extension.register_processor("pupil", {})
    extension.register_processor("whisking", {})
    extension.register_processor("accel", {})
    
    processors = extension.list_processors()
    assert "pupil" in processors
    assert "whisking" in processors
    assert "accel" in processors
    assert len(processors) == 3


def test_behavioral_analysis_extension_create_widget(qapp):
    """Test creating widget from extension."""
    extension = BehavioralAnalysisExtension()
    widget = extension.create_widget()
    
    assert widget is not None
    assert isinstance(widget, BehavioralAnalysisWidget)


def test_get_behavioral_extension():
    """Test getting global behavioral extension instance."""
    extension = get_behavioral_extension()
    assert extension is not None
    assert isinstance(extension, BehavioralAnalysisExtension)
    
    # Should return same instance
    extension2 = get_behavioral_extension()
    assert extension is extension2


def test_show_behavioral_analysis_info(qapp, monkeypatch):
    """Test showing behavioral analysis info dialog."""
    # Mock QMessageBox.information to avoid showing actual dialog
    called = []
    
    def mock_information(parent, title, message):
        called.append((parent, title, message))
    
    monkeypatch.setattr("ui.extensions.behavioral_analysis.QMessageBox.information", mock_information)
    
    show_behavioral_analysis_info()
    
    assert len(called) == 1
    assert "Behavioral" in called[0][1]  # Title should contain Behavioral


def test_behavioral_extension_multiple_processors():
    """Test behavioral extension with multiple processors."""
    extension = BehavioralAnalysisExtension()
    
    processor1 = {"type": "pupil"}
    processor2 = {"type": "whisking"}
    processor3 = {"type": "accel"}
    
    extension.register_processor("pupil", processor1)
    extension.register_processor("whisking", processor2)
    extension.register_processor("accel", processor3)
    
    assert len(extension.list_processors()) == 3
    
    # Activate multiple processors
    extension.activate_processor("pupil")
    extension.activate_processor("whisking")
    
    active = extension.list_active_processors()
    assert "pupil" in active
    assert "whisking" in active
    assert "accel" not in active
    assert len(active) == 2


def test_behavioral_widget_initial_state(qapp):
    """Test behavioral widget initial state."""
    widget = BehavioralAnalysisWidget()
    
    # Should have no data path initially
    assert widget._data_path is None
    
    # Should have no enabled data types initially
    assert len(widget._enabled_data_types) == 0
    
    # Status should indicate no data
    assert "no" in widget.status_label.text().lower() or \
           "not loaded" in widget.status_label.text().lower()


def test_behavioral_extension_prevent_duplicate_activation():
    """Test that activating same processor twice doesn't duplicate."""
    extension = BehavioralAnalysisExtension()
    
    extension.register_processor("test", {})
    extension.activate_processor("test")
    extension.activate_processor("test")  # Try to activate again
    
    active = extension.list_active_processors()
    assert active.count("test") == 1, "Processor should only appear once in active list"
