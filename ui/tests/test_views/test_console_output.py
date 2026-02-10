"""
Unit tests for ConsoleOutputWidget.

Tests output appending and auto-scroll behavior.
"""

import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.views.console_output_widget import ConsoleOutputWidget


@pytest.fixture
def app(qapp):
    """Provide QApplication instance."""
    return qapp


@pytest.fixture
def console_output(app):
    """Create a ConsoleOutputWidget instance for testing."""
    return ConsoleOutputWidget()


def test_text_area_present(console_output):
    """Test that text area is present."""
    assert console_output.text_area is not None
    assert console_output.text_area.isReadOnly()


def test_clear_button_present(console_output):
    """Test that clear button is present."""
    assert console_output.clear_button is not None
    assert "Clear" in console_output.clear_button.text()


def test_append_output(console_output):
    """Test appending text to output."""
    test_text = "Test output line\n"
    
    console_output.append_output(test_text)
    
    output = console_output.get_output()
    assert test_text in output


def test_append_multiple_outputs(console_output):
    """Test appending multiple lines of output."""
    lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
    
    for line in lines:
        console_output.append_output(line)
    
    output = console_output.get_output()
    for line in lines:
        assert line.strip() in output


def test_clear_output(console_output):
    """Test clearing output."""
    console_output.append_output("Some text\n")
    assert len(console_output.get_output()) > 0
    
    console_output.clear_output()
    assert len(console_output.get_output()) == 0


def test_clear_button_functionality(console_output, qtbot):
    """Test that clear button clears output."""
    console_output.append_output("Some text\n")
    assert len(console_output.get_output()) > 0
    
    console_output.clear_button.click()
    assert len(console_output.get_output()) == 0


def test_auto_scroll_enabled_by_default(console_output):
    """Test that auto-scroll is enabled by default."""
    assert console_output._auto_scroll is True


def test_auto_scroll_behavior(console_output, qtbot):
    """Test auto-scroll to bottom when appending output."""
    # Add enough text to require scrolling
    for i in range(100):
        console_output.append_output(f"Line {i}\n")
    
    # Process events to ensure scroll happens
    qtbot.wait(100)
    
    # Verify scrollbar is at bottom
    scrollbar = console_output.text_area.verticalScrollBar()
    assert scrollbar.value() == scrollbar.maximum()


def test_disable_auto_scroll(console_output, qtbot):
    """Test disabling auto-scroll."""
    console_output.set_auto_scroll(False)
    
    # Add some text
    console_output.append_output("Line 1\n")
    
    # Scroll to top
    scrollbar = console_output.text_area.verticalScrollBar()
    scrollbar.setValue(0)
    
    # Add more text
    for i in range(50):
        console_output.append_output(f"Line {i}\n")
    
    # Process events
    qtbot.wait(100)
    
    # Scrollbar should not be at bottom (auto-scroll disabled)
    assert scrollbar.value() < scrollbar.maximum()


def test_enable_auto_scroll(console_output, qtbot):
    """Test enabling auto-scroll."""
    console_output.set_auto_scroll(False)
    console_output.append_output("Line 1\n")
    
    # Re-enable auto-scroll
    console_output.set_auto_scroll(True)
    
    # Add more text
    for i in range(50):
        console_output.append_output(f"Line {i}\n")
    
    # Process events
    qtbot.wait(100)
    
    # Scrollbar should be at bottom
    scrollbar = console_output.text_area.verticalScrollBar()
    assert scrollbar.value() == scrollbar.maximum()


def test_scroll_to_bottom_method(console_output, qtbot):
    """Test manual scroll to bottom."""
    # Add text
    for i in range(100):
        console_output.append_output(f"Line {i}\n")
    
    # Scroll to top
    scrollbar = console_output.text_area.verticalScrollBar()
    scrollbar.setValue(0)
    
    # Manually scroll to bottom
    console_output.scroll_to_bottom()
    
    # Process events
    qtbot.wait(100)
    
    # Verify at bottom
    assert scrollbar.value() == scrollbar.maximum()


def test_get_output(console_output):
    """Test getting all output text."""
    test_lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
    
    for line in test_lines:
        console_output.append_output(line)
    
    output = console_output.get_output()
    
    for line in test_lines:
        assert line.strip() in output


def test_monospace_font(console_output):
    """Test that text area uses monospace font."""
    font = console_output.text_area.font()
    # Check if font family suggests monospace
    family = font.family().lower()
    assert "courier" in family or "mono" in family or font.fixedPitch()


def test_dark_theme_styling(console_output):
    """Test that console has dark theme styling."""
    stylesheet = console_output.text_area.styleSheet()
    # Should have dark background
    assert "#1e1e1e" in stylesheet or "1e1e1e" in stylesheet.lower()


def test_append_preserves_order(console_output):
    """Test that appending preserves order of output."""
    lines = [f"Line {i}\n" for i in range(10)]
    
    for line in lines:
        console_output.append_output(line)
    
    output = console_output.get_output()
    
    # Verify lines appear in order
    for i, line in enumerate(lines):
        assert line.strip() in output
        if i > 0:
            # Current line should appear after previous line
            prev_pos = output.find(lines[i-1].strip())
            curr_pos = output.find(line.strip())
            assert curr_pos > prev_pos


def test_append_empty_string(console_output):
    """Test appending empty string."""
    initial_output = console_output.get_output()
    
    console_output.append_output("")
    
    # Output should be unchanged
    assert console_output.get_output() == initial_output


def test_append_special_characters(console_output):
    """Test appending text with special characters."""
    special_text = "Test with special chars: \t\n\r!@#$%^&*()"
    
    console_output.append_output(special_text)
    
    output = console_output.get_output()
    # Most of the special text should be preserved
    assert "special chars" in output
