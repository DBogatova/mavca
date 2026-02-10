"""
Unit tests for PlotViewerWidget.

Tests plot loading, zoom/pan functionality, and plot type switching.
"""

import pytest
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import tempfile
import matplotlib.pyplot as plt

from ui.views.plot_viewer_widget import PlotViewerWidget


@pytest.fixture
def app(qapp):
    """Provide QApplication instance."""
    return qapp


@pytest.fixture
def plot_viewer(app):
    """Create a PlotViewerWidget instance for testing."""
    return PlotViewerWidget()


@pytest.fixture
def sample_plot_file():
    """Create a temporary plot image file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Plot")
        plt.savefig(f.name)
        plt.close(fig)
        
        yield Path(f.name)
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)


def test_figure_canvas_present(plot_viewer):
    """Test that matplotlib figure canvas is present."""
    assert plot_viewer.figure is not None
    assert plot_viewer.canvas is not None


def test_navigation_toolbar_present(plot_viewer):
    """Test that navigation toolbar for zoom/pan is present."""
    assert plot_viewer.toolbar is not None


def test_plot_type_selector_present(plot_viewer):
    """Test that plot type selector is present."""
    assert plot_viewer.plot_type_combo is not None
    
    # Check that it has the expected plot types
    items = [plot_viewer.plot_type_combo.itemText(i) 
             for i in range(plot_viewer.plot_type_combo.count())]
    
    assert "DFF Traces" in items
    assert "Depth Analysis" in items
    assert "Outside Mask Dynamics" in items


def test_refresh_button_present(plot_viewer):
    """Test that refresh button is present."""
    assert plot_viewer.refresh_button is not None
    assert "Refresh" in plot_viewer.refresh_button.text()


def test_status_label_present(plot_viewer):
    """Test that status label is present."""
    assert plot_viewer.status_label is not None


def test_load_plot_from_file(plot_viewer, sample_plot_file):
    """Test loading a plot from an image file."""
    success = plot_viewer.load_plot(sample_plot_file)
    
    assert success is True
    assert "Loaded" in plot_viewer.status_label.text()
    assert sample_plot_file.name in plot_viewer.status_label.text()


def test_load_nonexistent_plot(plot_viewer):
    """Test loading a plot from a nonexistent file."""
    nonexistent_path = Path("/nonexistent/plot.png")
    
    success = plot_viewer.load_plot(nonexistent_path)
    
    assert success is False
    assert "not found" in plot_viewer.status_label.text().lower()


def test_load_plot_data(plot_viewer):
    """Test loading plot from data arrays."""
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data)
    
    plot_viewer.load_plot_data(
        x_data, y_data,
        title="Sine Wave",
        xlabel="X",
        ylabel="Y"
    )
    
    assert "Loaded" in plot_viewer.status_label.text()
    assert "Sine Wave" in plot_viewer.status_label.text()


def test_plot_type_switching(plot_viewer, qtbot):
    """Test switching between plot types."""
    initial_type = plot_viewer.get_current_plot_type()
    
    # Change plot type
    plot_viewer.set_plot_type("Depth Analysis")
    
    # Process events
    qtbot.wait(100)
    
    new_type = plot_viewer.get_current_plot_type()
    assert new_type == "Depth Analysis"
    assert new_type != initial_type


def test_get_current_plot_type(plot_viewer):
    """Test getting current plot type."""
    current_type = plot_viewer.get_current_plot_type()
    
    assert current_type in ["DFF Traces", "Depth Analysis", "Outside Mask Dynamics"]


def test_set_plot_type(plot_viewer):
    """Test setting plot type."""
    plot_viewer.set_plot_type("Outside Mask Dynamics")
    
    assert plot_viewer.get_current_plot_type() == "Outside Mask Dynamics"


def test_refresh_button_with_no_plot(plot_viewer, qtbot):
    """Test refresh button when no plot is loaded."""
    plot_viewer.refresh_button.click()
    
    qtbot.wait(100)
    
    assert "No plot" in plot_viewer.status_label.text()


def test_refresh_button_with_plot(plot_viewer, sample_plot_file, qtbot):
    """Test refresh button with a loaded plot."""
    # Load plot
    plot_viewer.load_plot(sample_plot_file)
    
    # Click refresh
    plot_viewer.refresh_button.click()
    
    qtbot.wait(100)
    
    # Should still show loaded status
    assert "Loaded" in plot_viewer.status_label.text()


def test_zoom_pan_toolbar_functionality(plot_viewer):
    """Test that zoom/pan toolbar has expected actions."""
    toolbar = plot_viewer.toolbar
    
    # Check that toolbar has actions
    actions = toolbar.actions()
    assert len(actions) > 0
    
    # Toolbar should have zoom and pan capabilities
    # (NavigationToolbar2QT provides these by default)
    action_texts = [action.text() for action in actions if action.text()]
    
    # Check for common navigation actions
    has_navigation = any(
        text.lower() in ['zoom', 'pan', 'home', 'back', 'forward', 'save']
        for text in action_texts
    )
    assert has_navigation


def test_clear_plot(plot_viewer, sample_plot_file):
    """Test clearing the current plot."""
    # Load a plot
    plot_viewer.load_plot(sample_plot_file)
    assert plot_viewer._current_plot_path is not None
    
    # Clear plot
    plot_viewer._clear_plot()
    
    assert plot_viewer._current_plot_path is None


def test_plot_type_change_signal(plot_viewer, qtbot):
    """Test that changing plot type updates status."""
    # Change plot type via combo box
    plot_viewer.plot_type_combo.setCurrentText("Depth Analysis")
    
    qtbot.wait(100)
    
    # Status should reflect the selection
    assert "Depth Analysis" in plot_viewer.status_label.text()


def test_load_plot_with_title(plot_viewer):
    """Test loading plot data with title."""
    x_data = [1, 2, 3, 4, 5]
    y_data = [1, 4, 9, 16, 25]
    title = "Quadratic Function"
    
    plot_viewer.load_plot_data(x_data, y_data, title=title)
    
    # Title should appear in status
    assert title in plot_viewer.status_label.text()


def test_load_plot_with_labels(plot_viewer):
    """Test loading plot data with axis labels."""
    x_data = [1, 2, 3]
    y_data = [1, 2, 3]
    
    plot_viewer.load_plot_data(
        x_data, y_data,
        title="Test",
        xlabel="Time (s)",
        ylabel="Amplitude"
    )
    
    # Plot should be loaded successfully
    assert "Loaded" in plot_viewer.status_label.text()


def test_multiple_plot_loads(plot_viewer, sample_plot_file):
    """Test loading multiple plots sequentially."""
    # Load first plot
    success1 = plot_viewer.load_plot(sample_plot_file)
    assert success1
    
    # Load second plot (data)
    x_data = [1, 2, 3]
    y_data = [1, 2, 3]
    plot_viewer.load_plot_data(x_data, y_data, title="Second Plot")
    
    # Should show second plot
    assert "Second Plot" in plot_viewer.status_label.text()


def test_plot_viewer_initial_state(plot_viewer):
    """Test initial state of plot viewer."""
    # Should show "No plot loaded" initially
    assert "No plot" in plot_viewer.status_label.text().lower()
    
    # Default plot type should be set
    assert plot_viewer.get_current_plot_type() == "DFF Traces"


def test_error_handling_invalid_data(plot_viewer):
    """Test error handling with invalid plot data."""
    # Try to load with invalid data
    try:
        plot_viewer.load_plot_data(None, None, title="Invalid")
    except:
        pass
    
    # Should show error in status
    assert "error" in plot_viewer.status_label.text().lower() or "No plot" in plot_viewer.status_label.text()


def test_figure_updates_on_load(plot_viewer, sample_plot_file):
    """Test that figure is updated when plot is loaded."""
    # Get initial figure state
    initial_axes = len(plot_viewer.figure.get_axes())
    
    # Load plot
    plot_viewer.load_plot(sample_plot_file)
    
    # Figure should have axes now
    final_axes = len(plot_viewer.figure.get_axes())
    assert final_axes > 0
