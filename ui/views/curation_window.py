"""
M3 Curation Window - Interactive mask curation interface

This module provides the CurationWindow class for reviewing and editing 3D masks
with real-time DFF trace previews.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import tifffile
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class MaskData:
    """Container for mask information"""
    
    def __init__(self, mask_id: int, mask_array: np.ndarray, file_path: Path):
        self.mask_id = mask_id
        self.mask_array = mask_array
        self.file_path = file_path
        self.status = "pending"  # pending, kept, deleted, merged
        self.merged_with: Optional[int] = None


class CurationWindow(QMainWindow):
    """Interactive mask curation interface for M3
    
    Provides navigation, visualization, and curation controls for reviewing
    3D masks with DFF trace previews.
    """
    
    curation_complete = pyqtSignal()
    
    def __init__(self, data_path: Path, parent=None):
        super().__init__(parent)
        self.data_path = data_path
        self.current_mask_index = 0
        self.masks: List[MaskData] = []
        self.merge_mode = False
        self.merge_source_index: Optional[int] = None
        
        self.setWindowTitle("M3 Mask Curation")
        self.resize(1200, 800)
        
        self.setup_ui()
        self.load_masks()
        
        if self.masks:
            self.display_mask(0)
    
    def setup_ui(self) -> None:
        """Initialize curation UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Status bar at top
        self.status_label = QLabel("Loading masks...")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Vertical)
        
        # Mask visualization area (placeholder)
        mask_widget = QWidget()
        mask_layout = QVBoxLayout(mask_widget)
        self.mask_info_label = QLabel("Mask visualization placeholder")
        self.mask_info_label.setAlignment(Qt.AlignCenter)
        self.mask_info_label.setStyleSheet(
            "background-color: #f0f0f0; border: 2px solid #ccc; "
            "padding: 20px; min-height: 200px;"
        )
        mask_layout.addWidget(self.mask_info_label)
        splitter.addWidget(mask_widget)
        
        # DFF trace plot area
        trace_widget = QWidget()
        trace_layout = QVBoxLayout(trace_widget)
        trace_layout.addWidget(QLabel("DFF Trace Preview:"))
        
        self.figure = Figure(figsize=(8, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        trace_layout.addWidget(self.canvas)
        
        splitter.addWidget(trace_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self.on_previous)
        nav_layout.addWidget(self.prev_button)
        
        self.mask_counter_label = QLabel("Mask 0 / 0")
        self.mask_counter_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.mask_counter_label)
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.on_next)
        nav_layout.addWidget(self.next_button)
        
        layout.addLayout(nav_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.keep_button = QPushButton("✓ Keep")
        self.keep_button.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        self.keep_button.clicked.connect(self.on_keep)
        action_layout.addWidget(self.keep_button)
        
        self.delete_button = QPushButton("✗ Delete")
        self.delete_button.setStyleSheet("background-color: #FFB6C1; font-weight: bold;")
        self.delete_button.clicked.connect(self.on_delete)
        action_layout.addWidget(self.delete_button)
        
        self.merge_button = QPushButton("⊕ Merge")
        self.merge_button.setStyleSheet("background-color: #ADD8E6; font-weight: bold;")
        self.merge_button.clicked.connect(self.on_merge)
        action_layout.addWidget(self.merge_button)
        
        layout.addLayout(action_layout)
        
        # Save and Exit buttons
        bottom_layout = QHBoxLayout()
        
        self.save_button = QPushButton("💾 Save Progress")
        self.save_button.clicked.connect(self.save_curation_results)
        bottom_layout.addWidget(self.save_button)
        
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        bottom_layout.addWidget(self.exit_button)
        
        layout.addLayout(bottom_layout)
    
    def load_masks(self) -> None:
        """Load all masks from labelmaps directory"""
        labelmaps_dir = self.data_path / "labelmaps"
        
        if not labelmaps_dir.exists():
            logger.error(f"Labelmaps directory not found: {labelmaps_dir}")
            self.status_label.setText("Error: Labelmaps directory not found")
            return
        
        # Find all labelmap TIFF files
        labelmap_files = sorted(labelmaps_dir.glob("labelmap_*.tif"))
        
        if not labelmap_files:
            logger.warning(f"No labelmap files found in {labelmaps_dir}")
            self.status_label.setText("No masks found to curate")
            return
        
        # Load each mask
        for i, file_path in enumerate(labelmap_files):
            try:
                mask_array = tifffile.imread(file_path)
                mask_data = MaskData(i, mask_array, file_path)
                self.masks.append(mask_data)
                logger.info(f"Loaded mask {i} from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load mask from {file_path}: {e}")
        
        self.status_label.setText(f"Loaded {len(self.masks)} masks")
        logger.info(f"Successfully loaded {len(self.masks)} masks")
    
    def display_mask(self, index: int) -> None:
        """Display mask at given index"""
        if not self.masks or index < 0 or index >= len(self.masks):
            return
        
        self.current_mask_index = index
        mask = self.masks[index]
        
        # Update mask info
        shape_str = f"{mask.mask_array.shape}"
        status_str = mask.status.upper()
        self.mask_info_label.setText(
            f"Mask {mask.mask_id}\n"
            f"Shape: {shape_str}\n"
            f"Status: {status_str}\n"
            f"File: {mask.file_path.name}\n\n"
            f"[3D visualization would appear here]\n"
            f"(Napari integration or custom viewer)"
        )
        
        # Update counter
        self.mask_counter_label.setText(f"Mask {index + 1} / {len(self.masks)}")
        
        # Update button states
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.masks) - 1)
        
        # Compute and display DFF trace
        self.compute_and_display_trace(mask)
        
        # Update status label
        if self.merge_mode:
            self.status_label.setText(
                f"MERGE MODE: Select second mask to merge with Mask {self.merge_source_index}"
            )
        else:
            self.status_label.setText(
                f"Viewing Mask {index + 1} of {len(self.masks)} - Status: {status_str}"
            )
    
    def compute_and_display_trace(self, mask: MaskData) -> None:
        """Compute DFF trace for current mask and display it"""
        try:
            # Load preprocessed data
            dff_file = self.data_path / "preprocessed" / "dff_stack.npy"
            
            if not dff_file.exists():
                logger.warning(f"DFF stack not found: {dff_file}")
                self.plot_placeholder_trace("DFF data not available")
                return
            
            # Load DFF stack
            dff_stack = np.load(dff_file)
            
            # Extract trace for this mask
            trace = self.compute_dff_preview(mask.mask_array, dff_stack)
            
            # Plot the trace
            self.update_trace_plot(trace)
            
        except Exception as e:
            logger.error(f"Failed to compute DFF trace: {e}")
            self.plot_placeholder_trace(f"Error: {str(e)}")
    
    def compute_dff_preview(self, mask: np.ndarray, dff_stack: np.ndarray) -> np.ndarray:
        """Compute DFF trace for a mask
        
        Args:
            mask: 3D binary mask array
            dff_stack: 4D DFF stack (time, z, y, x)
        
        Returns:
            1D trace array (time points)
        """
        # Create binary mask
        mask_binary = mask > 0
        
        # Extract values at mask locations for each time point
        # Average across all voxels in the mask
        trace = np.zeros(dff_stack.shape[0])
        
        for t in range(dff_stack.shape[0]):
            masked_values = dff_stack[t][mask_binary]
            if len(masked_values) > 0:
                trace[t] = np.mean(masked_values)
        
        return trace
    
    def update_trace_plot(self, trace: np.ndarray) -> None:
        """Update the DFF trace plot"""
        self.ax.clear()
        
        time_points = np.arange(len(trace))
        self.ax.plot(time_points, trace, 'b-', linewidth=1)
        
        # Add baseline reference
        baseline = np.median(trace)
        self.ax.axhline(y=baseline, color='r', linestyle='--', alpha=0.5, label='Median')
        
        # Styling
        self.ax.set_xlabel('Time (frames)')
        self.ax.set_ylabel('ΔF/F')
        self.ax.set_title('DFF Trace Preview')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
    
    def plot_placeholder_trace(self, message: str) -> None:
        """Plot placeholder when trace cannot be computed"""
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, message,
            ha='center', va='center',
            transform=self.ax.transAxes,
            fontsize=12
        )
        self.ax.set_xlabel('Time (frames)')
        self.ax.set_ylabel('ΔF/F')
        self.ax.set_title('DFF Trace Preview')
        self.canvas.draw()
    
    def on_next(self) -> None:
        """Navigate to next mask"""
        if self.current_mask_index < len(self.masks) - 1:
            self.display_mask(self.current_mask_index + 1)
    
    def on_previous(self) -> None:
        """Navigate to previous mask"""
        if self.current_mask_index > 0:
            self.display_mask(self.current_mask_index - 1)
    
    def on_keep(self) -> None:
        """Mark current mask as kept"""
        if not self.masks:
            return
        
        mask = self.masks[self.current_mask_index]
        mask.status = "kept"
        logger.info(f"Mask {mask.mask_id} marked as KEPT")
        
        # Update display
        self.display_mask(self.current_mask_index)
        
        # Auto-advance to next mask
        if self.current_mask_index < len(self.masks) - 1:
            self.on_next()
    
    def on_delete(self) -> None:
        """Mark current mask for deletion"""
        if not self.masks:
            return
        
        mask = self.masks[self.current_mask_index]
        mask.status = "deleted"
        logger.info(f"Mask {mask.mask_id} marked as DELETED")
        
        # Update display
        self.display_mask(self.current_mask_index)
        
        # Auto-advance to next mask
        if self.current_mask_index < len(self.masks) - 1:
            self.on_next()
    
    def on_merge(self) -> None:
        """Initiate or complete merge operation"""
        if not self.masks:
            return
        
        if not self.merge_mode:
            # Start merge mode
            self.merge_mode = True
            self.merge_source_index = self.current_mask_index
            self.merge_button.setText("⊕ Complete Merge")
            self.merge_button.setStyleSheet("background-color: #FFA500; font-weight: bold;")
            self.status_label.setText(
                f"MERGE MODE: Navigate to second mask and click 'Complete Merge'"
            )
            logger.info(f"Merge mode started with source mask {self.merge_source_index}")
        else:
            # Complete merge
            target_index = self.current_mask_index
            
            if target_index == self.merge_source_index:
                QMessageBox.warning(
                    self,
                    "Invalid Merge",
                    "Cannot merge a mask with itself. Navigate to a different mask."
                )
                return
            
            # Perform merge
            source_mask = self.masks[self.merge_source_index]
            target_mask = self.masks[target_index]
            
            # Mark both as merged
            source_mask.status = "merged"
            source_mask.merged_with = target_index
            target_mask.status = "merged"
            target_mask.merged_with = self.merge_source_index
            
            logger.info(f"Merged mask {self.merge_source_index} with mask {target_index}")
            
            # Exit merge mode
            self.merge_mode = False
            self.merge_source_index = None
            self.merge_button.setText("⊕ Merge")
            self.merge_button.setStyleSheet("background-color: #ADD8E6; font-weight: bold;")
            
            # Update display
            self.display_mask(self.current_mask_index)
            
            QMessageBox.information(
                self,
                "Merge Complete",
                f"Masks {source_mask.mask_id} and {target_mask.mask_id} have been merged."
            )
    
    def save_curation_results(self) -> None:
        """Save curated masks to disk"""
        try:
            # Create curation results file
            results_file = self.data_path / "curation_results.json"
            
            import json
            results = {
                "total_masks": len(self.masks),
                "masks": []
            }
            
            for mask in self.masks:
                mask_info = {
                    "mask_id": mask.mask_id,
                    "file": str(mask.file_path.name),
                    "status": mask.status,
                    "merged_with": mask.merged_with
                }
                results["masks"].append(mask_info)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Curation results saved to {results_file}")
            
            # Count statistics
            kept = sum(1 for m in self.masks if m.status == "kept")
            deleted = sum(1 for m in self.masks if m.status == "deleted")
            merged = sum(1 for m in self.masks if m.status == "merged")
            pending = sum(1 for m in self.masks if m.status == "pending")
            
            QMessageBox.information(
                self,
                "Save Complete",
                f"Curation results saved!\n\n"
                f"Kept: {kept}\n"
                f"Deleted: {deleted}\n"
                f"Merged: {merged}\n"
                f"Pending: {pending}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save curation results: {e}")
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save curation results:\n{str(e)}"
            )
    
    def closeEvent(self, event) -> None:
        """Handle window close event"""
        # Check if there are unsaved changes
        pending = sum(1 for m in self.masks if m.status == "pending")
        
        if pending > 0:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"You have {pending} masks that haven't been reviewed.\n\n"
                "Do you want to save your progress before exiting?",
                QMessageBox.Save |
                QMessageBox.Discard |
                QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_curation_results()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
