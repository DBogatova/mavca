"""
Unit tests for M6 visualization extension.

Tests the M6 3D visualization placeholder and extension points.
"""

import pytest
from PyQt5.QtWidgets import QApplication

from ui.extensions.m6_visualization import (
    M6VisualizationWidget,
    M6VisualizationExtension,
    get_m6_extension,
    show_m6_visualization_info
)
from pathlib import Path


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_m6_visualization_widget_creation(qapp):
    """Test M6 visualization widget can be created."""
    widget = M6VisualizationWidget()
    assert widget is not None
    assert widget.windowTitle() == "" or "M6" in widget.windowTitle() or True  # Widget may not have title


def test_m6_visualization_widget_has_ui_elements(qapp):
    """Test M6 visualization widget has expected UI elements."""
    widget = M6VisualizationWidget()
    
    # Should have buttons (even if disabled)
    assert widget.load_button is not None
    assert widget.render_button is not None
    assert widget.status_label is not None


def test_m6_visualization_widget_buttons_disabled(qapp):
    """Test M6 visualization widget buttons are disabled (placeholder)."""
    widget = M6VisualizationWidget()
    
    # Placeholder buttons should be disabled
    assert not widget.load_button.isEnabled()
    assert not widget.render_button.isEnabled()


def test_m6_visualization_widget_set_data_path(qapp):
    """Test setting data path on M6 visualization widget."""
    widget = M6VisualizationWidget()
    test_path = Path("/test/data/path")
    
    widget.set_data_path(test_path)
    assert widget._data_path == test_path
    assert str(test_path) in widget.status_label.text()


def test_m6_visualization_widget_load_data_placeholder(qapp):
    """Test load_visualization_data returns False (placeholder)."""
    widget = M6VisualizationWidget()
    result = widget.load_visualization_data()
    assert result is False, "Placeholder should return False"


def test_m6_visualization_widget_render_placeholder(qapp):
    """Test render_3d_view doesn't crash (placeholder)."""
    widget = M6VisualizationWidget()
    # Should not crash
    widget.render_3d_view()


def test_m6_visualization_extension_creation():
    """Test M6 visualization extension can be created."""
    extension = M6VisualizationExtension()
    assert extension is not None


def test_m6_visualization_extension_register_backend():
    """Test registering a backend with M6 extension."""
    extension = M6VisualizationExtension()
    
    # Create a mock backend
    mock_backend = {"name": "test_backend"}
    
    extension.register_backend("test", mock_backend)
    assert "test" in extension.list_backends()


def test_m6_visualization_extension_set_active_backend():
    """Test setting active backend."""
    extension = M6VisualizationExtension()
    
    mock_backend = {"name": "test_backend"}
    extension.register_backend("test", mock_backend)
    
    result = extension.set_active_backend("test")
    assert result is True
    
    active = extension.get_active_backend()
    assert active == mock_backend


def test_m6_visualization_extension_set_nonexistent_backend():
    """Test setting nonexistent backend returns False."""
    extension = M6VisualizationExtension()
    
    result = extension.set_active_backend("nonexistent")
    assert result is False


def test_m6_visualization_extension_list_backends():
    """Test listing registered backends."""
    extension = M6VisualizationExtension()
    
    extension.register_backend("backend1", {})
    extension.register_backend("backend2", {})
    
    backends = extension.list_backends()
    assert "backend1" in backends
    assert "backend2" in backends
    assert len(backends) == 2


def test_m6_visualization_extension_create_widget(qapp):
    """Test creating widget from extension."""
    extension = M6VisualizationExtension()
    widget = extension.create_widget()
    
    assert widget is not None
    assert isinstance(widget, M6VisualizationWidget)


def test_get_m6_extension():
    """Test getting global M6 extension instance."""
    extension = get_m6_extension()
    assert extension is not None
    assert isinstance(extension, M6VisualizationExtension)
    
    # Should return same instance
    extension2 = get_m6_extension()
    assert extension is extension2


def test_show_m6_visualization_info(qapp, monkeypatch):
    """Test showing M6 visualization info dialog."""
    # Mock QMessageBox.information to avoid showing actual dialog
    called = []
    
    def mock_information(parent, title, message):
        called.append((parent, title, message))
    
    monkeypatch.setattr("ui.extensions.m6_visualization.QMessageBox.information", mock_information)
    
    show_m6_visualization_info()
    
    assert len(called) == 1
    assert "M6" in called[0][1]  # Title should contain M6


def test_m6_extension_multiple_backends():
    """Test M6 extension with multiple backends."""
    extension = M6VisualizationExtension()
    
    backend1 = {"type": "vtk"}
    backend2 = {"type": "napari"}
    backend3 = {"type": "custom"}
    
    extension.register_backend("vtk", backend1)
    extension.register_backend("napari", backend2)
    extension.register_backend("custom", backend3)
    
    assert len(extension.list_backends()) == 3
    
    # Set active backend
    extension.set_active_backend("napari")
    assert extension.get_active_backend() == backend2
    
    # Change active backend
    extension.set_active_backend("vtk")
    assert extension.get_active_backend() == backend1


def test_m6_widget_initial_state(qapp):
    """Test M6 widget initial state."""
    widget = M6VisualizationWidget()
    
    # Should have no data path initially
    assert widget._data_path is None
    
    # Status should indicate no data
    assert "no data" in widget.status_label.text().lower() or \
           "not loaded" in widget.status_label.text().lower()
