"""
Main window for the MAVCA Pipeline UI.

This module provides the MainWindow class that assembles all widgets
into the primary application interface.
"""

from typing import Optional
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMessageBox, QMenuBar, QMenu, QStatusBar
)
from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from PyQt5.QtWidgets import QAction
from PyQt5.QtGui import QKeySequence

from .parameter_panel import ParameterPanel
from .module_navigation_bar import ModuleNavigationBar
from .module_detail_view import ModuleDetailView
from .console_output_widget import ConsoleOutputWidget
from .plot_viewer_widget import PlotViewerWidget


class MainWindow(QMainWindow):
    """Main application window.
    
    Assembles all UI components and provides menu bar, keyboard shortcuts,
    and window geometry persistence.
    
    Signals:
        closing: Emitted when window is closing
    """
    
    closing = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the main window.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        self.settings = QSettings("MAVCA", "PipelineUI")
        
        # Widget references
        self.parameter_panel: Optional[ParameterPanel] = None
        self.navigation_bar: Optional[ModuleNavigationBar] = None
        self.module_detail_view: Optional[ModuleDetailView] = None
        self.console_output: Optional[ConsoleOutputWidget] = None
        self.plot_viewer: Optional[PlotViewerWidget] = None
        
        self._setup_ui()
        self._create_menu_bar()
        self._create_status_bar()
        self._setup_keyboard_shortcuts()
        self._restore_geometry()
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("MAVCA Pipeline UI")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Module navigation bar at top
        self.navigation_bar = ModuleNavigationBar()
        main_layout.addWidget(self.navigation_bar)
        
        # Horizontal splitter for main content
        h_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Parameters
        self.parameter_panel = ParameterPanel()
        self.parameter_panel.setMaximumWidth(350)
        h_splitter.addWidget(self.parameter_panel)
        
        # Center/Right: Vertical splitter
        v_splitter = QSplitter(Qt.Vertical)
        
        # Top: Module detail view
        self.module_detail_view = ModuleDetailView()
        v_splitter.addWidget(self.module_detail_view)
        
        # Middle: Console output
        self.console_output = ConsoleOutputWidget()
        v_splitter.addWidget(self.console_output)
        
        # Bottom: Plot viewer (initially hidden)
        self.plot_viewer = PlotViewerWidget()
        self.plot_viewer.setVisible(False)
        v_splitter.addWidget(self.plot_viewer)
        
        # Set initial sizes for vertical splitter
        v_splitter.setSizes([400, 200, 200])
        
        h_splitter.addWidget(v_splitter)
        
        # Set initial sizes for horizontal splitter
        h_splitter.setSizes([300, 900])
        
        main_layout.addWidget(h_splitter)
    
    def _create_menu_bar(self) -> None:
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # New dataset action
        new_action = QAction("&New Dataset...", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setStatusTip("Configure a new dataset")
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Toggle console action
        self.toggle_console_action = QAction("Show &Console", self)
        self.toggle_console_action.setCheckable(True)
        self.toggle_console_action.setChecked(True)
        self.toggle_console_action.setShortcut("Ctrl+Shift+C")
        self.toggle_console_action.setStatusTip("Toggle console output visibility")
        self.toggle_console_action.triggered.connect(self._toggle_console)
        view_menu.addAction(self.toggle_console_action)
        
        # Toggle plot viewer action
        self.toggle_plot_action = QAction("Show &Plot Viewer", self)
        self.toggle_plot_action.setCheckable(True)
        self.toggle_plot_action.setChecked(False)
        self.toggle_plot_action.setShortcut("Ctrl+Shift+P")
        self.toggle_plot_action.setStatusTip("Toggle plot viewer visibility")
        self.toggle_plot_action.triggered.connect(self._toggle_plot_viewer)
        view_menu.addAction(self.toggle_plot_action)
        
        view_menu.addSeparator()
        
        # Clear console action
        clear_console_action = QAction("Clear C&onsole", self)
        clear_console_action.setShortcut("Ctrl+L")
        clear_console_action.setStatusTip("Clear console output")
        clear_console_action.triggered.connect(self._clear_console)
        view_menu.addAction(clear_console_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.setStatusTip("About MAVCA Pipeline UI")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        # Documentation action
        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        docs_action.setStatusTip("Open documentation")
        help_menu.addAction(docs_action)
    
    def _create_status_bar(self) -> None:
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _setup_keyboard_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        # Module navigation shortcuts (Ctrl+1 through Ctrl+8)
        for i, module_id in enumerate(["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"], 1):
            if i <= 9:  # Only create shortcuts for 1-9
                action = QAction(self)
                action.setShortcut(f"Ctrl+{i}")
                action.triggered.connect(
                    lambda checked, mid=module_id: self._navigate_to_module(mid)
                )
                self.addAction(action)
    
    def _navigate_to_module(self, module_id: str) -> None:
        """Navigate to a specific module.
        
        Args:
            module_id: Module identifier to navigate to
        """
        if self.navigation_bar:
            self.navigation_bar.set_current_module(module_id)
            self.navigation_bar.module_selected.emit(module_id)
    
    def _toggle_console(self, checked: bool) -> None:
        """Toggle console output visibility.
        
        Args:
            checked: True to show, False to hide
        """
        if self.console_output:
            self.console_output.setVisible(checked)
    
    def _toggle_plot_viewer(self, checked: bool) -> None:
        """Toggle plot viewer visibility.
        
        Args:
            checked: True to show, False to hide
        """
        if self.plot_viewer:
            self.plot_viewer.setVisible(checked)
    
    def _clear_console(self) -> None:
        """Clear the console output."""
        if self.console_output:
            self.console_output.clear_output()
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About MAVCA Pipeline UI",
            "<h2>MAVCA Pipeline UI</h2>"
            "<p>Mask-Assisted Volumetric Calcium Analysis Pipeline</p>"
            "<p>A graphical interface for processing 4D calcium imaging data.</p>"
            "<p>Version 1.0</p>"
        )
    
    def _restore_geometry(self) -> None:
        """Restore window geometry from settings."""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        window_state = self.settings.value("windowState")
        if window_state:
            self.restoreState(window_state)
    
    def _save_geometry(self) -> None:
        """Save window geometry to settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    def closeEvent(self, event) -> None:
        """Handle window close event.
        
        Args:
            event: Close event
        """
        self._save_geometry()
        self.closing.emit()
        event.accept()
    
    def show_error(self, title: str, message: str) -> None:
        """Display an error dialog.
        
        Args:
            title: Error dialog title
            message: Error message
        """
        QMessageBox.critical(self, title, message)
    
    def show_warning(self, title: str, message: str) -> None:
        """Display a warning dialog.
        
        Args:
            title: Warning dialog title
            message: Warning message
        """
        QMessageBox.warning(self, title, message)
    
    def show_info(self, title: str, message: str) -> None:
        """Display an information dialog.
        
        Args:
            title: Info dialog title
            message: Info message
        """
        QMessageBox.information(self, title, message)
    
    def ask_question(self, title: str, message: str) -> bool:
        """Display a yes/no question dialog.
        
        Args:
            title: Question dialog title
            message: Question message
            
        Returns:
            True if user clicked Yes, False otherwise
        """
        reply = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes
    
    def update_status(self, message: str) -> None:
        """Update the status bar message.
        
        Args:
            message: Status message to display
        """
        if self.status_bar:
            self.status_bar.showMessage(message)
    
    def append_console_output(self, text: str) -> None:
        """Append text to console output.
        
        Args:
            text: Text to append
        """
        if self.console_output:
            self.console_output.append_output(text)
    
    def get_parameter_panel(self) -> ParameterPanel:
        """Get the parameter panel widget.
        
        Returns:
            ParameterPanel instance
        """
        return self.parameter_panel
    
    def get_navigation_bar(self) -> ModuleNavigationBar:
        """Get the navigation bar widget.
        
        Returns:
            ModuleNavigationBar instance
        """
        return self.navigation_bar
    
    def get_module_detail_view(self) -> ModuleDetailView:
        """Get the module detail view widget.
        
        Returns:
            ModuleDetailView instance
        """
        return self.module_detail_view
    
    def get_console_output(self) -> ConsoleOutputWidget:
        """Get the console output widget.
        
        Returns:
            ConsoleOutputWidget instance
        """
        return self.console_output
    
    def get_plot_viewer(self) -> PlotViewerWidget:
        """Get the plot viewer widget.
        
        Returns:
            PlotViewerWidget instance
        """
        return self.plot_viewer
    def show_plot_viewer(self) -> None:
        """Show the plot viewer panel."""
        if self.plot_viewer:
            self.plot_viewer.setVisible(True)
