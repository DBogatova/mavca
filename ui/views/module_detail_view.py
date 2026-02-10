"""
Module detail view widget for the MAVCA Pipeline UI.

This module provides the ModuleDetailView widget for displaying
detailed information about a pipeline module.
"""

from pathlib import Path
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from ..models.module_definition import ModuleDefinition


class ModuleDetailView(QWidget):
    """Widget for displaying module details.
    
    Shows module name, description, scripts, expected outputs,
    and execution controls.
    
    Signals:
        execute_module: Emitted when execute button is clicked (module_id)
        open_directory: Emitted when open directory button is clicked (module_id)
    """
    
    execute_module = pyqtSignal(str)
    open_directory = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the module detail view.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        self._current_module: Optional[ModuleDefinition] = None
        self._output_existence: Dict[str, bool] = {}
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Main layout with scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        # Content widget
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Module header
        self.module_title = QLabel()
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.module_title.setFont(title_font)
        self.content_layout.addWidget(self.module_title)
        
        self.module_description = QLabel()
        self.module_description.setWordWrap(True)
        self.module_description.setStyleSheet("color: #666666; font-size: 12pt;")
        self.content_layout.addWidget(self.module_description)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.execute_button = QPushButton("Execute Module")
        self.execute_button.setMinimumHeight(40)
        self.execute_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.execute_button.clicked.connect(self._on_execute_clicked)
        button_layout.addWidget(self.execute_button)
        
        self.open_dir_button = QPushButton("Open Directory")
        self.open_dir_button.setMinimumHeight(40)
        self.open_dir_button.clicked.connect(self._on_open_dir_clicked)
        button_layout.addWidget(self.open_dir_button)
        
        self.content_layout.addLayout(button_layout)
        
        # Scripts section
        self.scripts_group = QGroupBox("Scripts")
        self.scripts_layout = QVBoxLayout()
        self.scripts_group.setLayout(self.scripts_layout)
        self.content_layout.addWidget(self.scripts_group)
        
        # Expected outputs section
        self.outputs_group = QGroupBox("Expected Outputs")
        self.outputs_layout = QVBoxLayout()
        self.outputs_group.setLayout(self.outputs_layout)
        self.content_layout.addWidget(self.outputs_group)
        
        # Add stretch
        self.content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
    
    def set_module(self, module: ModuleDefinition) -> None:
        """Set the module to display.
        
        Args:
            module: Module definition to display
        """
        self._current_module = module
        self._update_display()
    
    def set_output_existence(self, existence: Dict[str, bool]) -> None:
        """Set which expected outputs exist.
        
        Args:
            existence: Dictionary mapping output paths to existence status
        """
        self._output_existence = existence.copy()
        self._update_outputs_display()
    
    def set_execution_enabled(self, enabled: bool) -> None:
        """Enable or disable the execute button.
        
        Args:
            enabled: True to enable, False to disable
        """
        self.execute_button.setEnabled(enabled)
    
    def _update_display(self) -> None:
        """Update the display with current module information."""
        if not self._current_module:
            self.module_title.setText("No module selected")
            self.module_description.setText("")
            self._clear_scripts()
            self._clear_outputs()
            return
        
        module = self._current_module
        
        # Update header
        title = f"{module.id}: {module.name}"
        if module.optional:
            title += " (Optional)"
        if module.requires_interaction:
            title += " [Interactive]"
        
        self.module_title.setText(title)
        self.module_description.setText(module.description)
        
        # Update scripts
        self._update_scripts_display()
        
        # Update outputs
        self._update_outputs_display()
    
    def _update_scripts_display(self) -> None:
        """Update the scripts section."""
        self._clear_scripts()
        
        if not self._current_module:
            return
        
        for script in self._current_module.scripts:
            # Script container
            script_frame = QFrame()
            script_frame.setFrameShape(QFrame.Shape.StyledPanel)
            script_frame.setStyleSheet("""
                QFrame {
                    background-color: #f9f9f9;
                    border: 1px solid #e0e0e0;
                    border-radius: 3px;
                    padding: 5px;
                }
            """)
            
            script_layout = QVBoxLayout(script_frame)
            
            # Script name
            name_label = QLabel(script.name)
            name_font = QFont()
            name_font.setBold(True)
            name_label.setFont(name_font)
            if script.optional:
                name_label.setText(f"{script.name} (Optional)")
                name_label.setStyleSheet("color: #666666;")
            script_layout.addWidget(name_label)
            
            # Script description
            desc_label = QLabel(script.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #555555; font-size: 10pt;")
            script_layout.addWidget(desc_label)
            
            # Script parameters (if any)
            if script.parameters:
                params_text = "Parameters: " + ", ".join(
                    f"{k}={v}" for k, v in script.parameters.items()
                )
                params_label = QLabel(params_text)
                params_label.setWordWrap(True)
                params_label.setStyleSheet("color: #0078d4; font-size: 9pt; font-family: monospace;")
                script_layout.addWidget(params_label)
                
                # Categorization note
                cat_label = QLabel("(Common: DATE, MOUSE, RUN | Script-specific: shown above)")
                cat_label.setStyleSheet("color: #999999; font-size: 8pt; font-style: italic;")
                script_layout.addWidget(cat_label)
            
            self.scripts_layout.addWidget(script_frame)
    
    def _update_outputs_display(self) -> None:
        """Update the expected outputs section."""
        self._clear_outputs()
        
        if not self._current_module:
            return
        
        has_missing = False
        
        for output_path in self._current_module.expected_outputs:
            # Output container
            output_layout = QHBoxLayout()
            
            # Existence indicator
            exists = self._output_existence.get(output_path, False)
            if exists:
                indicator = QLabel("✓")
                indicator.setStyleSheet("color: green; font-size: 14pt; font-weight: bold;")
            else:
                indicator = QLabel("✗")
                indicator.setStyleSheet("color: red; font-size: 14pt; font-weight: bold;")
                has_missing = True
            
            indicator.setFixedWidth(30)
            output_layout.addWidget(indicator)
            
            # Output path
            path_label = QLabel(output_path)
            path_label.setWordWrap(True)
            path_label.setStyleSheet("font-family: monospace; font-size: 10pt;")
            output_layout.addWidget(path_label, 1)
            
            self.outputs_layout.addLayout(output_layout)
        
        # Add warning if outputs are missing
        if has_missing and self._output_existence:
            warning = QLabel("⚠ Some expected outputs are missing")
            warning.setStyleSheet("""
                color: #ff6b6b;
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 3px;
                padding: 5px;
                font-weight: bold;
            """)
            self.outputs_layout.addWidget(warning)
    
    def _clear_scripts(self) -> None:
        """Clear the scripts section."""
        while self.scripts_layout.count():
            item = self.scripts_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _clear_outputs(self) -> None:
        """Clear the outputs section."""
        while self.outputs_layout.count():
            item = self.outputs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Clear nested layout
                while item.layout().count():
                    nested_item = item.layout().takeAt(0)
                    if nested_item.widget():
                        nested_item.widget().deleteLater()
    
    def _on_execute_clicked(self) -> None:
        """Handle execute button click."""
        if self._current_module:
            self.execute_module.emit(self._current_module.id)
    
    def _on_open_dir_clicked(self) -> None:
        """Handle open directory button click."""
        if self._current_module:
            self.open_directory.emit(self._current_module.id)
