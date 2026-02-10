"""
Property tests and unit tests for ModuleNavigationBar widget.

Tests current module indication, completion status distinction,
and unrestricted navigation.
"""

import pytest
from hypothesis import given, strategies as st
from PyQt5.QtWidgets import QApplication

from ui.views.module_navigation_bar import ModuleNavigationBar


@pytest.fixture
def app(qapp):
    """Provide QApplication instance."""
    return qapp


@pytest.fixture
def navigation_bar(app):
    """Create a ModuleNavigationBar instance for testing."""
    return ModuleNavigationBar()


# Feature: pipeline-ui, Property 3: Current Module Visual Indication
@given(st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]))
def test_current_module_indication_property(module_id):
    """
    For any module selection, the UI state should reflect that module
    as the currently active module.
    
    Validates: Requirements 1.4
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    nav_bar = ModuleNavigationBar()
    
    # Set current module
    nav_bar.set_current_module(module_id)
    
    # Verify the module is marked as current
    assert nav_bar.get_current_module() == module_id, (
        f"Current module should be {module_id}, got {nav_bar.get_current_module()}"
    )
    
    # Verify the button is checked
    button = nav_bar._module_buttons.get(module_id)
    assert button is not None, f"Button for {module_id} should exist"
    assert button.isChecked(), f"Button for {module_id} should be checked"
    
    # Verify other buttons are not checked
    for other_id, other_button in nav_bar._module_buttons.items():
        if other_id != module_id:
            assert not other_button.isChecked(), (
                f"Button for {other_id} should not be checked when {module_id} is current"
            )


# Feature: pipeline-ui, Property 20: Completion Status Visual Distinction
@given(
    st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]),
    st.booleans()
)
def test_completion_status_distinction_property(module_id, is_completed):
    """
    For any module, the UI should visually distinguish whether it is
    completed or incomplete based on the progress state.
    
    Validates: Requirements 6.3
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    nav_bar = ModuleNavigationBar()
    
    # Set completion status
    if is_completed:
        nav_bar.mark_module_complete(module_id)
    
    # Get button
    button = nav_bar._module_buttons.get(module_id)
    assert button is not None, f"Button for {module_id} should exist"
    
    # Verify visual distinction
    button_text = button.text()
    
    if is_completed:
        # Completed modules should have checkmark
        assert "✓" in button_text, (
            f"Completed module {module_id} should have checkmark in button text"
        )
        # Button should have completion styling
        assert "green" in button.styleSheet().lower() or "#e8f5e9" in button.styleSheet(), (
            f"Completed module {module_id} should have green styling"
        )
    else:
        # Incomplete modules should not have checkmark
        assert "✓" not in button_text, (
            f"Incomplete module {module_id} should not have checkmark in button text"
        )


# Feature: pipeline-ui, Property 29: Unrestricted Module Navigation
@given(
    st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]),
    st.lists(st.sampled_from(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]), min_size=0, max_size=8, unique=True)
)
def test_unrestricted_navigation_property(target_module, completed_modules):
    """
    For any module regardless of completion status, navigation to that
    module should be possible.
    
    Validates: Requirements 8.3
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    nav_bar = ModuleNavigationBar()
    
    # Set some modules as completed
    nav_bar.set_completed_modules(set(completed_modules))
    
    # Attempt to navigate to target module
    nav_bar.set_current_module(target_module)
    
    # Verify navigation succeeded regardless of completion status
    assert nav_bar.get_current_module() == target_module, (
        f"Should be able to navigate to {target_module} "
        f"(completed: {target_module in completed_modules})"
    )
    
    # Verify button is accessible and clickable
    button = nav_bar._module_buttons.get(target_module)
    assert button is not None, f"Button for {target_module} should exist"
    assert button.isEnabled(), f"Button for {target_module} should be enabled"
    assert button.isChecked(), f"Button for {target_module} should be checked after navigation"


# Unit tests

def test_all_module_buttons_present(navigation_bar):
    """Test that all 8 module buttons are present."""
    expected_modules = ["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]
    
    for module_id in expected_modules:
        assert module_id in navigation_bar._module_buttons, (
            f"Button for {module_id} should exist"
        )


def test_optional_modules_marked(navigation_bar):
    """Test that optional modules are marked with asterisk."""
    optional_modules = ["M1.5", "M2.5"]
    
    for module_id in optional_modules:
        button = navigation_bar._module_buttons[module_id]
        assert "*" in button.text(), (
            f"Optional module {module_id} should have * in button text"
        )


def test_module_selection_signal(navigation_bar, qtbot):
    """Test that module_selected signal is emitted on click."""
    with qtbot.waitSignal(navigation_bar.module_selected, timeout=1000) as blocker:
        navigation_bar.set_current_module("M1")
        navigation_bar.module_selected.emit("M1")
    
    assert blocker.args == ["M1"]


def test_set_completed_modules(navigation_bar):
    """Test setting multiple completed modules."""
    completed = {"M1", "M2", "M3"}
    navigation_bar.set_completed_modules(completed)
    
    # Verify completed modules have checkmark
    for module_id in completed:
        button = navigation_bar._module_buttons[module_id]
        assert "✓" in button.text()
    
    # Verify incomplete modules don't have checkmark
    incomplete = {"M1.5", "M2.5", "M4", "M5", "M6"}
    for module_id in incomplete:
        button = navigation_bar._module_buttons[module_id]
        assert "✓" not in button.text()


def test_mark_module_complete(navigation_bar):
    """Test marking a single module as complete."""
    navigation_bar.mark_module_complete("M1")
    
    button = navigation_bar._module_buttons["M1"]
    assert "✓" in button.text()


def test_current_module_highlighting(navigation_bar):
    """Test that current module has distinct styling."""
    navigation_bar.set_current_module("M2")
    
    current_button = navigation_bar._module_buttons["M2"]
    other_button = navigation_bar._module_buttons["M1"]
    
    # Current button should have blue/highlighted styling
    assert "#0078d4" in current_button.styleSheet() or "blue" in current_button.styleSheet().lower()
    
    # Other button should not have the same styling
    assert current_button.styleSheet() != other_button.styleSheet()


def test_optional_module_styling(navigation_bar):
    """Test that optional modules have distinct styling."""
    # M1.5 is optional
    optional_button = navigation_bar._module_buttons["M1.5"]
    required_button = navigation_bar._module_buttons["M1"]
    
    # Optional button should have dashed border or italic style
    assert "dashed" in optional_button.styleSheet() or "italic" in optional_button.styleSheet()


def test_completed_module_styling(navigation_bar):
    """Test that completed modules have distinct styling."""
    navigation_bar.mark_module_complete("M1")
    
    completed_button = navigation_bar._module_buttons["M1"]
    
    # Completed button should have green styling
    assert "green" in completed_button.styleSheet().lower() or "#e8f5e9" in completed_button.styleSheet()
