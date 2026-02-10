"""
M2 completion dialog for the MAVCA Pipeline UI.

This module provides a dialog that appears after M2 completion,
offering options to proceed to M3 or refine with M2.5.
"""

from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QGroupBox, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt


class M2CompletionDialog(QDialog):
    """Dialog shown after M2 completion.
    
    Provides options to:
    - Proceed directly to M3 curation
    - Run M2.5 to add manual guides for refinement
    - Compare guided and non-guided labelmaps (if both exist)
    
    Attributes:
        selected_action: The action selected by the user
        has_guided: Whether guided labelmaps exist
        has_non_guided: Whether non-guided labelmaps exist
    """
    
    ACTION_PROCEED_M3 = "proceed_m3"
    ACTION_RUN_M2_5 = "run_m2_5"
    ACTION_COMPARE = "compare"
    
    def __init__(
        self,
        has_guided: bool = False,
        has_non_guided: bool = True,
        parent: Optional[QWidget] = None
    ):
        """Initialize the M2 completion dialog.
        
        Args:
            has_guided: Whether guided labelmaps exist
            has_non_guided: Whether non-guided labelmaps exist
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        self.has_guided = has_guided
        self.has_non_guided = has_non_guided
        self.selected_action: Optional[str] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("M2 Completed Successfully")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Title and description
        title_label = QLabel("<h2>M2: Initial Mask Creation Complete</h2>")
        layout.addWidget(title_label)
        
        description = QLabel(
            "M2 has successfully created initial dendrite masks. "
            "You can now proceed to M3 for interactive curation, "
            "or optionally run M2.5 to add manual trunk guides for "
            "improved segmentation quality."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        layout.addSpacing(20)
        
        # Options group
        options_group = QGroupBox("What would you like to do next?")
        options_layout = QVBoxLayout(options_group)
        
        self.button_group = QButtonGroup(self)
        
        # Option 1: Proceed to M3
        self.proceed_radio = QRadioButton(
            "Proceed to M3 (Interactive Mask Curation)"
        )
        self.proceed_radio.setChecked(True)
        self.button_group.addButton(self.proceed_radio)
        options_layout.addWidget(self.proceed_radio)
        
        proceed_desc = QLabel(
            "   Continue with the current masks and start interactive curation."
        )
        proceed_desc.setStyleSheet("color: gray; font-size: 10pt;")
        proceed_desc.setWordWrap(True)
        options_layout.addWidget(proceed_desc)
        
        options_layout.addSpacing(10)
        
        # Option 2: Run M2.5
        self.refine_radio = QRadioButton(
            "Run M2.5 (Draw Manual Trunk Guides)"
        )
        self.button_group.addButton(self.refine_radio)
        options_layout.addWidget(self.refine_radio)
        
        refine_desc = QLabel(
            "   Draw manual guides to improve segmentation, then re-run M2 with guides enabled."
        )
        refine_desc.setStyleSheet("color: gray; font-size: 10pt;")
        refine_desc.setWordWrap(True)
        options_layout.addWidget(refine_desc)
        
        # Option 3: Compare (only if both guided and non-guided exist)
        if self.has_guided and self.has_non_guided:
            options_layout.addSpacing(10)
            
            self.compare_radio = QRadioButton(
                "Compare Guided and Non-Guided Labelmaps"
            )
            self.button_group.addButton(self.compare_radio)
            options_layout.addWidget(self.compare_radio)
            
            compare_desc = QLabel(
                "   View both guided and non-guided labelmaps side-by-side to decide which to use."
            )
            compare_desc.setStyleSheet("color: gray; font-size: 10pt;")
            compare_desc.setWordWrap(True)
            options_layout.addWidget(compare_desc)
        
        layout.addWidget(options_group)
        
        # Guided labelmap indicator
        if self.has_guided:
            layout.addSpacing(10)
            guided_label = QLabel(
                "ℹ️ <b>Note:</b> Guided labelmaps detected. "
                "M2 was run with manual trunk guides from M2.5."
            )
            guided_label.setStyleSheet("color: #0066cc; padding: 10px; background-color: #e6f2ff; border-radius: 5px;")
            guided_label.setWordWrap(True)
            layout.addWidget(guided_label)
        
        layout.addSpacing(20)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        ok_button = QPushButton("Continue")
        ok_button.setDefault(True)
        ok_button.clicked.connect(self._on_continue)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
    
    def _on_continue(self) -> None:
        """Handle continue button click."""
        if self.proceed_radio.isChecked():
            self.selected_action = self.ACTION_PROCEED_M3
        elif self.refine_radio.isChecked():
            self.selected_action = self.ACTION_RUN_M2_5
        elif hasattr(self, 'compare_radio') and self.compare_radio.isChecked():
            self.selected_action = self.ACTION_COMPARE
        else:
            self.selected_action = self.ACTION_PROCEED_M3
        
        self.accept()
    
    def get_selected_action(self) -> Optional[str]:
        """Get the action selected by the user.
        
        Returns:
            One of ACTION_PROCEED_M3, ACTION_RUN_M2_5, ACTION_COMPARE, or None
        """
        return self.selected_action


class M2_5CompletionDialog(QDialog):
    """Dialog shown after M2.5 completion.
    
    Informs the user that manual guides have been created and offers
    to re-run M2 with guides enabled.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the M2.5 completion dialog.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("M2.5 Completed Successfully")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Title and description
        title_label = QLabel("<h2>M2.5: Manual Trunk Guides Created</h2>")
        layout.addWidget(title_label)
        
        description = QLabel(
            "M2.5 has successfully created manual trunk guides. "
            "You should now re-run M2 to create improved labelmaps "
            "using these guides. The guided segmentation will use your "
            "manual trunk annotations to produce higher quality masks."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        layout.addSpacing(20)
        
        # Info box
        info_label = QLabel(
            "ℹ️ <b>Next Step:</b> Navigate to M2 and run it again. "
            "The script will automatically detect and use your manual guides."
        )
        info_label.setStyleSheet("color: #0066cc; padding: 10px; background-color: #e6f2ff; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addSpacing(20)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_button = QPushButton("OK")
        ok_button.setDefault(True)
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
