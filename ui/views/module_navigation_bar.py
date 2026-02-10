"""
Module navigation bar widget for the MAVCA Pipeline UI.

This module provides the ModuleNavigationBar widget for navigating
between pipeline modules.
"""

from typing import Optional, Set

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QButtonGroup, QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt


class ModuleNavigationBar(QWidget):
    """Widget for navigating between pipeline modules.
    
    Provides buttons for each module with visual indication of:
    - Current module (highlighted)
    - Completed modules (checkmark)
    - Optional modules (marked with *)
    
    Signals:
        module_selected: Emitted when a module is selected (module_id)
    """
    
    module_selected = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the module navigation bar.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        self._current_module: Optional[str] = None
        self._completed_modules: Set[str] = set()
        self._optional_modules: Set[str] = {"M1.5", "M2.5"}
        self._module_buttons: dict[str, QPushButton] = {}
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create button group for exclusive selection
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(False)  # Allow manual control
        
        # Create buttons for all modules
        module_ids = ["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]
        
        for module_id in module_ids:
            button = QPushButton(self._format_button_text(module_id))
            button.setCheckable(True)
            button.setMinimumWidth(80)
            button.setMinimumHeight(40)
            button.clicked.connect(lambda checked, mid=module_id: self._on_module_clicked(mid))
            
            self.button_group.addButton(button)
            self._module_buttons[module_id] = button
            layout.addWidget(button)
            
            # Update button style
            self._update_button_style(module_id)
        
        layout.addStretch()
    
    def _format_button_text(self, module_id: str) -> str:
        """Format button text with optional indicator.
        
        Args:
            module_id: Module identifier
            
        Returns:
            Formatted button text
        """
        text = module_id
        if module_id in self._optional_modules:
            text += " *"
        return text
    
    def _on_module_clicked(self, module_id: str) -> None:
        """Handle module button click.
        
        Args:
            module_id: Clicked module identifier
        """
        self.set_current_module(module_id)
        self.module_selected.emit(module_id)
    
    def set_current_module(self, module_id: str) -> None:
        """Set the current active module.
        
        Args:
            module_id: Module identifier to set as current
        """
        self._current_module = module_id
        
        # Update all button states
        for mid, button in self._module_buttons.items():
            button.setChecked(mid == module_id)
            self._update_button_style(mid)
    
    def set_completed_modules(self, completed: Set[str]) -> None:
        """Set which modules are completed.
        
        Args:
            completed: Set of completed module identifiers
        """
        self._completed_modules = completed.copy()
        
        # Update all button styles
        for module_id in self._module_buttons:
            self._update_button_style(module_id)
    
    def mark_module_complete(self, module_id: str) -> None:
        """Mark a module as completed.
        
        Args:
            module_id: Module identifier to mark as complete
        """
        self._completed_modules.add(module_id)
        self._update_button_style(module_id)
    
    def _update_button_style(self, module_id: str) -> None:
        """Update the visual style of a module button.
        
        Args:
            module_id: Module identifier
        """
        button = self._module_buttons.get(module_id)
        if not button:
            return
        
        is_current = module_id == self._current_module
        is_completed = module_id in self._completed_modules
        is_optional = module_id in self._optional_modules
        
        # Build button text with completion indicator
        text = module_id
        if is_optional:
            text += " *"
        if is_completed:
            text = "✓ " + text
        
        button.setText(text)
        
        # Apply styling
        if is_current:
            # Current module: highlighted
            button.setStyleSheet("""
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: 2px solid #005a9e;
                    border-radius: 5px;
                    font-weight: bold;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
            """)
        elif is_completed:
            # Completed module: green tint
            button.setStyleSheet("""
                QPushButton {
                    background-color: #e8f5e9;
                    color: #2e7d32;
                    border: 1px solid #4caf50;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #c8e6c9;
                }
            """)
        elif is_optional:
            # Optional module: lighter style
            button.setStyleSheet("""
                QPushButton {
                    background-color: #f5f5f5;
                    color: #666666;
                    border: 1px dashed #cccccc;
                    border-radius: 5px;
                    padding: 5px;
                    font-style: italic;
                }
                QPushButton:hover {
                    background-color: #eeeeee;
                }
            """)
        else:
            # Default style
            button.setStyleSheet("""
                QPushButton {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                }
            """)
    
    def get_current_module(self) -> Optional[str]:
        """Get the currently selected module.
        
        Returns:
            Current module identifier or None
        """
        return self._current_module
