"""
Plot viewer widget for the MAVCA Pipeline UI.

This module provides the PlotViewerWidget for displaying analysis plots
from M4 and M5 modules.
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotViewerWidget(QWidget):
    """Widget for viewing analysis plots.
    
    Embeds matplotlib figures with zoom/pan controls and plot type selection.
    Supports viewing:
    - DFF traces
    - Depth analysis plots
    - Outside mask dynamics plots
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the plot viewer widget.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        self._current_plot_path: Optional[Path] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        # Title
        title = QLabel("Plot Viewer")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Plot type selector
        type_label = QLabel("Plot Type:")
        header_layout.addWidget(type_label)
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "DFF Traces",
            "Depth Analysis",
            "Outside Mask Dynamics"
        ])
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        header_layout.addWidget(self.plot_type_combo)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        header_layout.addWidget(self.refresh_button)
        
        layout.addLayout(header_layout)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Navigation toolbar for zoom/pan
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        # Status label
        self.status_label = QLabel("No plot loaded")
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self.status_label)
    
    def load_plot(self, plot_path: Path) -> bool:
        """Load and display a plot from a file.
        
        Supports PNG, SVG, and PDF files.
        
        Args:
            plot_path: Path to the plot image file
            
        Returns:
            True if plot loaded successfully, False otherwise
        """
        if not plot_path.exists():
            self.status_label.setText(f"Plot not found: {plot_path.name}")
            self.status_label.setStyleSheet("color: red;")
            self._clear_plot()
            return False
        
        try:
            # Clear previous plot
            self.figure.clear()
            
            suffix = plot_path.suffix.lower()
            
            if suffix == ".pdf":
                # Render PDF using matplotlib's PdfPages reader
                from matplotlib.backends.backend_pdf import PdfPages
                import matplotlib.image as mpimg
                
                # Convert PDF to image via matplotlib
                # Read the PDF by rendering it
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(str(plot_path), first_page=1, last_page=1, dpi=150)
                    import numpy as np
                    img = np.array(images[0])
                except ImportError:
                    # Fallback: use subprocess to convert with sips (macOS)
                    import subprocess
                    import tempfile
                    import numpy as np
                    
                    # Convert PDF to PNG using sips (macOS built-in)
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    subprocess.run(
                        ['sips', '-s', 'format', 'png', str(plot_path), '--out', tmp_path],
                        capture_output=True, timeout=10
                    )
                    
                    if Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
                        img = plt.imread(tmp_path)
                        Path(tmp_path).unlink(missing_ok=True)
                    else:
                        Path(tmp_path).unlink(missing_ok=True)
                        raise RuntimeError("PDF conversion failed")
                
                ax = self.figure.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
            else:
                # Load PNG/SVG/other image formats
                img = plt.imread(str(plot_path))
                ax = self.figure.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self._current_plot_path = plot_path
            self.status_label.setText(f"Loaded: {plot_path.name}")
            self.status_label.setStyleSheet("color: green;")
            
            return True
            
        except Exception as e:
            self.status_label.setText(f"Error loading plot: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            self._clear_plot()
            return False
    
    def load_plot_data(self, x_data, y_data, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
        """Load and display plot from data arrays.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        try:
            # Clear previous plot
            self.figure.clear()
            
            # Create plot
            ax = self.figure.add_subplot(111)
            ax.plot(x_data, y_data)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            ax.grid(True, alpha=0.3)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self.status_label.setText(f"Loaded: {title}")
            self.status_label.setStyleSheet("color: green;")
            
        except Exception as e:
            self.status_label.setText(f"Error creating plot: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            self._clear_plot()
    
    def _clear_plot(self) -> None:
        """Clear the current plot."""
        self.figure.clear()
        self.canvas.draw()
        self._current_plot_path = None
    
    def _on_plot_type_changed(self, plot_type: str) -> None:
        """Handle plot type selection change.
        
        Args:
            plot_type: Selected plot type
        """
        # This would be connected to controller to load appropriate plot
        self.status_label.setText(f"Selected: {plot_type}")
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")
    
    def _on_refresh_clicked(self) -> None:
        """Handle refresh button click."""
        if self._current_plot_path:
            self.load_plot(self._current_plot_path)
        else:
            self.status_label.setText("No plot to refresh")
            self.status_label.setStyleSheet("color: #666666; font-style: italic;")
    
    def get_current_plot_type(self) -> str:
        """Get the currently selected plot type.
        
        Returns:
            Current plot type string
        """
        return self.plot_type_combo.currentText()
    
    def set_plot_type(self, plot_type: str) -> None:
        """Set the plot type selector.
        
        Args:
            plot_type: Plot type to select
        """
        index = self.plot_type_combo.findText(plot_type)
        if index >= 0:
            self.plot_type_combo.setCurrentIndex(index)
