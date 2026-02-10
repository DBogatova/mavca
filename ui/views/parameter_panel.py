"""
Parameter panel widget for the MAVCA Pipeline UI.

This module provides the ParameterPanel widget for configuring
DATE, MOUSE, RUN parameters and base directory.
"""

from pathlib import Path
from typing import Optional, Callable

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QGroupBox, QFormLayout
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont


class ParameterPanel(QWidget):
    """Widget for configuring pipeline parameters.
    
    Provides input fields for DATE, MOUSE, RUN parameters with validation
    indicators and a base directory configuration button.
    
    Signals:
        parameters_changed: Emitted when any parameter changes (date, mouse, run)
        base_dir_changed: Emitted when base directory changes (path)
    """
    
    parameters_changed = pyqtSignal(str, str, str)  # date, mouse, run
    base_dir_changed = pyqtSignal(Path)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the parameter panel.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        # Validation callbacks (set by controller)
        self.validate_date_callback: Optional[Callable[[str], bool]] = None
        self.validate_identifier_callback: Optional[Callable[[str], bool]] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Pipeline Parameters")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Parameters group box
        params_group = QGroupBox("Common Parameters")
        params_layout = QFormLayout()
        params_group.setLayout(params_layout)
        
        # DATE field
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("YYYY-MM-DD")
        self.date_input.textChanged.connect(self._on_date_text_changed)
        self.date_input.editingFinished.connect(self._on_date_editing_finished)
        self.date_validation_label = QLabel()
        self.date_validation_label.setStyleSheet("color: red; font-size: 10px;")
        date_container = QVBoxLayout()
        date_container.addWidget(self.date_input)
        date_container.addWidget(self.date_validation_label)
        date_container.setContentsMargins(0, 0, 0, 0)
        date_widget = QWidget()
        date_widget.setLayout(date_container)
        params_layout.addRow("DATE:", date_widget)
        
        # MOUSE field
        self.mouse_input = QLineEdit()
        self.mouse_input.setPlaceholderText("e.g., rAi162_phpeb")
        self.mouse_input.textChanged.connect(self._on_mouse_text_changed)
        self.mouse_input.editingFinished.connect(self._on_mouse_editing_finished)
        self.mouse_validation_label = QLabel()
        self.mouse_validation_label.setStyleSheet("color: red; font-size: 10px;")
        mouse_container = QVBoxLayout()
        mouse_container.addWidget(self.mouse_input)
        mouse_container.addWidget(self.mouse_validation_label)
        mouse_container.setContentsMargins(0, 0, 0, 0)
        mouse_widget = QWidget()
        mouse_widget.setLayout(mouse_container)
        params_layout.addRow("MOUSE:", mouse_widget)
        
        # RUN field
        self.run_input = QLineEdit()
        self.run_input.setPlaceholderText("e.g., run1")
        self.run_input.textChanged.connect(self._on_run_text_changed)
        self.run_input.editingFinished.connect(self._on_run_editing_finished)
        self.run_validation_label = QLabel()
        self.run_validation_label.setStyleSheet("color: red; font-size: 10px;")
        run_container = QVBoxLayout()
        run_container.addWidget(self.run_input)
        run_container.addWidget(self.run_validation_label)
        run_container.setContentsMargins(0, 0, 0, 0)
        run_widget = QWidget()
        run_widget.setLayout(run_container)
        params_layout.addRow("RUN:", run_widget)
        
        layout.addWidget(params_group)
        
        # Parameter change indicator
        self.change_indicator = QLabel()
        self.change_indicator.setStyleSheet("color: blue; font-style: italic;")
        self.change_indicator.setWordWrap(True)
        layout.addWidget(self.change_indicator)
        
        # Base directory group box
        base_dir_group = QGroupBox("Base Directory")
        base_dir_layout = QVBoxLayout()
        base_dir_group.setLayout(base_dir_layout)
        
        # Base directory display and button
        self.base_dir_label = QLabel()
        self.base_dir_label.setWordWrap(True)
        self.base_dir_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        base_dir_layout.addWidget(self.base_dir_label)
        
        self.base_dir_button = QPushButton("Change Base Directory...")
        self.base_dir_button.clicked.connect(self._on_base_dir_clicked)
        base_dir_layout.addWidget(self.base_dir_button)
        
        layout.addWidget(base_dir_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def _on_date_text_changed(self, text: str) -> None:
        """Handle DATE input text changes (on every keystroke).
        
        Args:
            text: New date text
        """
        # Clear validation error while typing
        self.date_validation_label.setText("")
        self.date_input.setStyleSheet("")
        
        # Show change indicator
        if text:
            self._show_change_indicator()
    
    def _on_date_editing_finished(self) -> None:
        """Handle DATE input editing finished (on focus loss or Enter)."""
        text = self.date_input.text()
        
        # Validate if callback is set
        if self.validate_date_callback and text:
            is_valid = self.validate_date_callback(text)
            if not is_valid:
                self.date_validation_label.setText("Invalid format. Use YYYY-MM-DD")
                self.date_input.setStyleSheet("border: 1px solid red;")
        
        # Emit signal
        self._emit_parameters_changed()
    
    def _on_mouse_text_changed(self, text: str) -> None:
        """Handle MOUSE input text changes (on every keystroke).
        
        Args:
            text: New mouse text
        """
        # Clear validation error while typing
        self.mouse_validation_label.setText("")
        self.mouse_input.setStyleSheet("")
        
        # Show change indicator
        if text:
            self._show_change_indicator()
    
    def _on_mouse_editing_finished(self) -> None:
        """Handle MOUSE input editing finished (on focus loss or Enter)."""
        text = self.mouse_input.text()
        
        # Validate if callback is set
        if self.validate_identifier_callback and text:
            is_valid = self.validate_identifier_callback(text)
            if not is_valid:
                self.mouse_validation_label.setText("Only letters, numbers, underscore, hyphen")
                self.mouse_input.setStyleSheet("border: 1px solid red;")
        
        # Emit signal
        self._emit_parameters_changed()
    
    def _on_run_text_changed(self, text: str) -> None:
        """Handle RUN input text changes (on every keystroke).
        
        Args:
            text: New run text
        """
        # Clear validation error while typing
        self.run_validation_label.setText("")
        self.run_input.setStyleSheet("")
        
        # Show change indicator
        if text:
            self._show_change_indicator()
    
    def _on_run_editing_finished(self) -> None:
        """Handle RUN input editing finished (on focus loss or Enter)."""
        text = self.run_input.text()
        
        # Validate if callback is set
        if self.validate_identifier_callback and text:
            is_valid = self.validate_identifier_callback(text)
            if not is_valid:
                self.run_validation_label.setText("Only letters, numbers, underscore, hyphen")
                self.run_input.setStyleSheet("border: 1px solid red;")
        
        # Emit signal
        self._emit_parameters_changed()
    
    def _show_change_indicator(self) -> None:
        """Show indicator that parameters have changed."""
        self.change_indicator.setText(
            "✓ Parameters updated. New values will be used on next execution."
        )
    
    def _emit_parameters_changed(self) -> None:
        """Emit parameters_changed signal with current values."""
        date = self.date_input.text()
        mouse = self.mouse_input.text()
        run = self.run_input.text()
        self.parameters_changed.emit(date, mouse, run)
    
    def _on_base_dir_clicked(self) -> None:
        """Handle base directory button click."""
        current_dir = self.base_dir_label.text() or str(Path.home())
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Base Data Directory",
            current_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directory:
            path = Path(directory)
            self.set_base_dir(path)
            self.base_dir_changed.emit(path)
    
    def set_date(self, date: str) -> None:
        """Set the DATE value.
        
        Args:
            date: Date string to set
        """
        self.date_input.setText(date)
    
    def set_mouse(self, mouse: str) -> None:
        """Set the MOUSE value.
        
        Args:
            mouse: Mouse identifier to set
        """
        self.mouse_input.setText(mouse)
    
    def set_run(self, run: str) -> None:
        """Set the RUN value.
        
        Args:
            run: Run identifier to set
        """
        self.run_input.setText(run)
    
    def set_base_dir(self, path: Path) -> None:
        """Set the base directory display.
        
        Args:
            path: Base directory path
        """
        self.base_dir_label.setText(str(path))
    
    def get_date(self) -> str:
        """Get the current DATE value.
        
        Returns:
            Current date string
        """
        return self.date_input.text()
    
    def get_mouse(self) -> str:
        """Get the current MOUSE value.
        
        Returns:
            Current mouse identifier
        """
        return self.mouse_input.text()
    
    def get_run(self) -> str:
        """Get the current RUN value.
        
        Returns:
            Current run identifier
        """
        return self.run_input.text()
    
    def clear_change_indicator(self) -> None:
        """Clear the parameter change indicator."""
        self.change_indicator.setText("")
