"""
Property tests and unit tests for MainWindow.

Tests navigation preserves configuration, window geometry persistence,
and all UI elements presence.
"""

import pytest
from hypothesis import given, strategies as st
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings
import tempfile

from ui.views.main_window import MainWindow
from ui.models.pipeline_config import PipelineConfig


@pytest.fixture
def app(qapp):
    """Provide QApplication instance."""
    return qapp


@pytest.fixture
def main_window(app):
    """Create a MainWindow instance for testing."""
    # Use temporary settings
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = Path(tmpdir) / "test_settings.ini"
        window = MainWindow()
        window.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        yield window
        window.close()


# Feature: pipeline-ui, Property 30: Navigation Preserves Configuration
@given(
    st.lists(st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]), min_size=2, max_size=8, unique=True),
    st.text(min_size=10, max_size=10).filter(lambda s: bool(__import__('re').match(r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$', s))),
    st.text(min_size=1, max_size=20).filter(lambda s: bool(__import__('re').match(r'^[a-zA-Z0-9_-]+$', s))),
    st.text(min_size=1, max_size=20).filter(lambda s: bool(__import__('re').match(r'^[a-zA-Z0-9_-]+$', s)))
)
def test_navigation_preserves_configuration_property(module_sequence, date, mouse, run):
    """
    For any sequence of module navigations, the parameter configuration
    should remain unchanged.
    
    Validates: Requirements 8.5
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create window
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = Path(tmpdir) / "test_settings.ini"
        window = MainWindow()
        window.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        
        # Set initial configuration
        param_panel = window.get_parameter_panel()
        param_panel.set_date(date)
        param_panel.set_mouse(mouse)
        param_panel.set_run(run)
        
        # Store initial values
        initial_date = param_panel.get_date()
        initial_mouse = param_panel.get_mouse()
        initial_run = param_panel.get_run()
        
        # Navigate through modules
        nav_bar = window.get_navigation_bar()
        for module_id in module_sequence:
            nav_bar.set_current_module(module_id)
        
        # Verify configuration unchanged
        assert param_panel.get_date() == initial_date, (
            f"DATE changed during navigation: {initial_date} -> {param_panel.get_date()}"
        )
        assert param_panel.get_mouse() == initial_mouse, (
            f"MOUSE changed during navigation: {initial_mouse} -> {param_panel.get_mouse()}"
        )
        assert param_panel.get_run() == initial_run, (
            f"RUN changed during navigation: {initial_run} -> {param_panel.get_run()}"
        )
        
        window.close()


# Feature: pipeline-ui, Property 33: Window Geometry Persistence Round-Trip
@given(
    st.integers(min_value=100, max_value=2000),
    st.integers(min_value=100, max_value=2000),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
)
def test_window_geometry_persistence_property(width, height, x, y):
    """
    For any window geometry (position and size), saving then loading
    should produce equivalent geometry values.
    
    Validates: Requirements 10.7
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = Path(tmpdir) / "test_settings.ini"
        
        # Create first window and set geometry
        window1 = MainWindow()
        window1.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        window1.resize(width, height)
        window1.move(x, y)
        
        # Save geometry
        window1._save_geometry()
        
        # Get saved geometry
        saved_geometry = window1.saveGeometry()
        
        window1.close()
        
        # Create second window and restore geometry
        window2 = MainWindow()
        window2.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        window2._restore_geometry()
        
        # Get restored geometry
        restored_geometry = window2.saveGeometry()
        
        # Verify geometries match
        assert saved_geometry == restored_geometry, (
            "Window geometry should be preserved across save/load"
        )
        
        window2.close()


# Unit tests

def test_parameter_panel_present(main_window):
    """Test that parameter panel is present."""
    param_panel = main_window.get_parameter_panel()
    assert param_panel is not None


def test_navigation_bar_present(main_window):
    """Test that module navigation bar is present."""
    nav_bar = main_window.get_navigation_bar()
    assert nav_bar is not None


def test_module_detail_view_present(main_window):
    """Test that module detail view is present."""
    detail_view = main_window.get_module_detail_view()
    assert detail_view is not None


def test_console_output_present(main_window):
    """Test that console output widget is present."""
    console = main_window.get_console_output()
    assert console is not None


def test_plot_viewer_present(main_window):
    """Test that plot viewer widget is present."""
    plot_viewer = main_window.get_plot_viewer()
    assert plot_viewer is not None


def test_menu_bar_present(main_window):
    """Test that menu bar is present."""
    menubar = main_window.menuBar()
    assert menubar is not None


def test_file_menu_present(main_window):
    """Test that File menu is present."""
    menubar = main_window.menuBar()
    actions = menubar.actions()
    
    menu_titles = [action.text() for action in actions]
    assert any("File" in title for title in menu_titles)


def test_view_menu_present(main_window):
    """Test that View menu is present."""
    menubar = main_window.menuBar()
    actions = menubar.actions()
    
    menu_titles = [action.text() for action in actions]
    assert any("View" in title for title in menu_titles)


def test_help_menu_present(main_window):
    """Test that Help menu is present."""
    menubar = main_window.menuBar()
    actions = menubar.actions()
    
    menu_titles = [action.text() for action in actions]
    assert any("Help" in title for title in menu_titles)


def test_status_bar_present(main_window):
    """Test that status bar is present."""
    status_bar = main_window.statusBar()
    assert status_bar is not None


def test_keyboard_shortcut_module_navigation(main_window):
    """Test keyboard shortcuts for module navigation."""
    # Check that actions with Ctrl+1 through Ctrl+8 exist
    actions = main_window.actions()
    
    shortcuts = [action.shortcut().toString() for action in actions if not action.shortcut().isEmpty()]
    
    # Should have shortcuts for module navigation
    assert any("Ctrl+1" in shortcut for shortcut in shortcuts)


def test_keyboard_shortcut_console_toggle(main_window):
    """Test keyboard shortcut for toggling console."""
    assert main_window.toggle_console_action is not None
    assert "Ctrl+Shift+C" in main_window.toggle_console_action.shortcut().toString()


def test_keyboard_shortcut_plot_toggle(main_window):
    """Test keyboard shortcut for toggling plot viewer."""
    assert main_window.toggle_plot_action is not None
    assert "Ctrl+Shift+P" in main_window.toggle_plot_action.shortcut().toString()


def test_keyboard_shortcut_clear_console(main_window):
    """Test keyboard shortcut for clearing console."""
    # Find clear console action
    menubar = main_window.menuBar()
    view_menu = None
    for action in menubar.actions():
        if "View" in action.text():
            view_menu = action.menu()
            break
    
    assert view_menu is not None
    
    # Check for clear console action with Ctrl+L
    actions = view_menu.actions()
    shortcuts = [action.shortcut().toString() for action in actions if not action.shortcut().isEmpty()]
    assert any("Ctrl+L" in shortcut for shortcut in shortcuts)


def test_exit_menu_item(main_window):
    """Test that Exit menu item exists."""
    menubar = main_window.menuBar()
    file_menu = None
    for action in menubar.actions():
        if "File" in action.text():
            file_menu = action.menu()
            break
    
    assert file_menu is not None
    
    actions = file_menu.actions()
    action_texts = [action.text() for action in actions]
    assert any("Exit" in text or "Quit" in text for text in action_texts)


def test_about_menu_item(main_window):
    """Test that About menu item exists."""
    menubar = main_window.menuBar()
    help_menu = None
    for action in menubar.actions():
        if "Help" in action.text():
            help_menu = action.menu()
            break
    
    assert help_menu is not None
    
    actions = help_menu.actions()
    action_texts = [action.text() for action in actions]
    assert any("About" in text for text in action_texts)


def test_documentation_menu_item(main_window):
    """Test that Documentation menu item exists."""
    menubar = main_window.menuBar()
    help_menu = None
    for action in menubar.actions():
        if "Help" in action.text():
            help_menu = action.menu()
            break
    
    assert help_menu is not None
    
    actions = help_menu.actions()
    action_texts = [action.text() for action in actions]
    assert any("Documentation" in text for text in action_texts)


def test_show_error_dialog(main_window, qtbot):
    """Test showing error dialog."""
    # This will show a dialog, so we need to close it automatically
    def close_dialog():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, type(main_window).__bases__[0]):
                if widget != main_window:
                    widget.close()
    
    # Schedule dialog close
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(100, close_dialog)
    
    # Show error
    main_window.show_error("Test Error", "This is a test error message")


def test_show_warning_dialog(main_window, qtbot):
    """Test showing warning dialog."""
    def close_dialog():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, type(main_window).__bases__[0]):
                if widget != main_window:
                    widget.close()
    
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(100, close_dialog)
    
    main_window.show_warning("Test Warning", "This is a test warning message")


def test_show_info_dialog(main_window, qtbot):
    """Test showing info dialog."""
    def close_dialog():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, type(main_window).__bases__[0]):
                if widget != main_window:
                    widget.close()
    
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(100, close_dialog)
    
    main_window.show_info("Test Info", "This is a test info message")


def test_update_status(main_window):
    """Test updating status bar message."""
    test_message = "Test status message"
    main_window.update_status(test_message)
    
    assert test_message in main_window.statusBar().currentMessage()


def test_append_console_output(main_window):
    """Test appending text to console output."""
    test_text = "Test console output\n"
    main_window.append_console_output(test_text)
    
    console = main_window.get_console_output()
    assert test_text.strip() in console.get_output()


def test_toggle_console_visibility(main_window, qtbot):
    """Test toggling console visibility."""
    console = main_window.get_console_output()
    
    # Initially visible
    assert console.isVisible()
    
    # Toggle off
    main_window.toggle_console_action.trigger()
    qtbot.wait(100)
    assert not console.isVisible()
    
    # Toggle on
    main_window.toggle_console_action.trigger()
    qtbot.wait(100)
    assert console.isVisible()


def test_toggle_plot_viewer_visibility(main_window, qtbot):
    """Test toggling plot viewer visibility."""
    plot_viewer = main_window.get_plot_viewer()
    
    # Initially hidden
    assert not plot_viewer.isVisible()
    
    # Toggle on
    main_window.toggle_plot_action.trigger()
    qtbot.wait(100)
    assert plot_viewer.isVisible()
    
    # Toggle off
    main_window.toggle_plot_action.trigger()
    qtbot.wait(100)
    assert not plot_viewer.isVisible()


def test_window_title(main_window):
    """Test that window has correct title."""
    assert "MAVCA" in main_window.windowTitle()
    assert "Pipeline" in main_window.windowTitle()


def test_minimum_window_size(main_window):
    """Test that window has minimum size set."""
    min_size = main_window.minimumSize()
    assert min_size.width() >= 1000
    assert min_size.height() >= 600


def test_closing_signal(main_window, qtbot):
    """Test that closing signal is emitted on window close."""
    with qtbot.waitSignal(main_window.closing, timeout=1000):
        main_window.close()
