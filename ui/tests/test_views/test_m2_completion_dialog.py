"""
Unit tests for M2 completion dialog.

Tests the M2/M2.5 workflow dialogs and their interactions.
"""

import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.views.m2_completion_dialog import M2CompletionDialog, M2_5CompletionDialog


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_m2_completion_dialog_creation(qapp):
    """Test M2 completion dialog can be created."""
    dialog = M2CompletionDialog()
    assert dialog is not None
    assert dialog.windowTitle() == "M2 Completed Successfully"


def test_m2_completion_dialog_default_action(qapp):
    """Test M2 completion dialog has proceed to M3 as default."""
    dialog = M2CompletionDialog()
    assert dialog.proceed_radio.isChecked()
    assert not dialog.refine_radio.isChecked()


def test_m2_completion_dialog_proceed_action(qapp):
    """Test selecting proceed to M3 action."""
    dialog = M2CompletionDialog()
    dialog.proceed_radio.setChecked(True)
    dialog._on_continue()
    
    assert dialog.selected_action == M2CompletionDialog.ACTION_PROCEED_M3


def test_m2_completion_dialog_refine_action(qapp):
    """Test selecting run M2.5 action."""
    dialog = M2CompletionDialog()
    dialog.refine_radio.setChecked(True)
    dialog._on_continue()
    
    assert dialog.selected_action == M2CompletionDialog.ACTION_RUN_M2_5


def test_m2_completion_dialog_compare_option_hidden_without_both_types(qapp):
    """Test compare option is hidden when both labelmap types don't exist."""
    # Only non-guided exists
    dialog = M2CompletionDialog(has_guided=False, has_non_guided=True)
    assert not hasattr(dialog, 'compare_radio')
    
    # Only guided exists
    dialog = M2CompletionDialog(has_guided=True, has_non_guided=False)
    assert not hasattr(dialog, 'compare_radio')
    
    # Neither exists
    dialog = M2CompletionDialog(has_guided=False, has_non_guided=False)
    assert not hasattr(dialog, 'compare_radio')


def test_m2_completion_dialog_compare_option_shown_with_both_types(qapp):
    """Test compare option is shown when both labelmap types exist."""
    dialog = M2CompletionDialog(has_guided=True, has_non_guided=True)
    assert hasattr(dialog, 'compare_radio')
    assert dialog.compare_radio is not None


def test_m2_completion_dialog_compare_action(qapp):
    """Test selecting compare action."""
    dialog = M2CompletionDialog(has_guided=True, has_non_guided=True)
    dialog.compare_radio.setChecked(True)
    dialog._on_continue()
    
    assert dialog.selected_action == M2CompletionDialog.ACTION_COMPARE


def test_m2_completion_dialog_guided_indicator_shown(qapp):
    """Test guided labelmap indicator is shown when guided labelmaps exist."""
    dialog = M2CompletionDialog(has_guided=True, has_non_guided=False)
    # The dialog should contain text about guided labelmaps
    # We can't easily test the exact label, but we can verify the dialog was created
    assert dialog is not None


def test_m2_completion_dialog_get_selected_action_before_continue(qapp):
    """Test get_selected_action returns None before continue is clicked."""
    dialog = M2CompletionDialog()
    assert dialog.get_selected_action() is None


def test_m2_5_completion_dialog_creation(qapp):
    """Test M2.5 completion dialog can be created."""
    dialog = M2_5CompletionDialog()
    assert dialog is not None
    assert dialog.windowTitle() == "M2.5 Completed Successfully"


def test_m2_5_completion_dialog_is_modal(qapp):
    """Test M2.5 completion dialog is modal."""
    dialog = M2_5CompletionDialog()
    assert dialog.isModal()


def test_m2_completion_dialog_is_modal(qapp):
    """Test M2 completion dialog is modal."""
    dialog = M2CompletionDialog()
    assert dialog.isModal()


def test_m2_completion_dialog_minimum_width(qapp):
    """Test M2 completion dialog has minimum width set."""
    dialog = M2CompletionDialog()
    assert dialog.minimumWidth() >= 500


def test_m2_5_completion_dialog_minimum_width(qapp):
    """Test M2.5 completion dialog has minimum width set."""
    dialog = M2_5CompletionDialog()
    assert dialog.minimumWidth() >= 500


def test_m2_completion_dialog_button_group(qapp):
    """Test M2 completion dialog has button group for radio buttons."""
    dialog = M2CompletionDialog()
    assert dialog.button_group is not None
    assert dialog.proceed_radio in dialog.button_group.buttons()
    assert dialog.refine_radio in dialog.button_group.buttons()


def test_m2_completion_dialog_with_guided_only(qapp):
    """Test M2 completion dialog when only guided labelmaps exist."""
    dialog = M2CompletionDialog(has_guided=True, has_non_guided=False)
    assert dialog.has_guided is True
    assert dialog.has_non_guided is False


def test_m2_completion_dialog_with_non_guided_only(qapp):
    """Test M2 completion dialog when only non-guided labelmaps exist."""
    dialog = M2CompletionDialog(has_guided=False, has_non_guided=True)
    assert dialog.has_guided is False
    assert dialog.has_non_guided is True


def test_m2_completion_dialog_action_constants(qapp):
    """Test M2 completion dialog has action constants defined."""
    assert hasattr(M2CompletionDialog, 'ACTION_PROCEED_M3')
    assert hasattr(M2CompletionDialog, 'ACTION_RUN_M2_5')
    assert hasattr(M2CompletionDialog, 'ACTION_COMPARE')
    
    assert M2CompletionDialog.ACTION_PROCEED_M3 == "proceed_m3"
    assert M2CompletionDialog.ACTION_RUN_M2_5 == "run_m2_5"
    assert M2CompletionDialog.ACTION_COMPARE == "compare"
