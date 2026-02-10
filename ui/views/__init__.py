"""
View components for the MAVCA Pipeline UI.

This package contains all PyQt6 widgets and UI components.
"""

from .parameter_panel import ParameterPanel
from .console_output_widget import ConsoleOutputWidget
from .module_navigation_bar import ModuleNavigationBar
from .module_detail_view import ModuleDetailView
from .plot_viewer_widget import PlotViewerWidget
from .main_window import MainWindow

__all__ = [
    "ParameterPanel",
    "ConsoleOutputWidget",
    "ModuleNavigationBar",
    "ModuleDetailView",
    "PlotViewerWidget",
    "MainWindow",
]
