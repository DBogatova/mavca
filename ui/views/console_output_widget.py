"""
Console output widget for the MAVCA Pipeline UI.

This module provides the ConsoleOutputWidget for displaying real-time
script execution output with a progress bar.
"""

import re
from typing import Optional, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout,
    QLabel, QProgressBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor


# Known progress steps for each module's scripts
# Maps step keywords to (step_number, total_steps) for progress calculation
MODULE_STEPS = {
    "find_events_m1": [
        "Loading stack",
        "Normalizing voxels",
        "Subtracting mean",
        "Applying Gaussian",
        "Computing rolling baseline",
        "Detecting events",
        "Creating activity timeline",
        "Grouping consecutive",
        "Saving event crops",
        "Saving processed stacks",
        "Module 1 processing complete",
    ],
    "pre_segmentation": [
        "Loading",
        "Computing",
        "Segmenting",
        "Saving",
    ],
    "detect_masks": [
        "Loading",
        "Processing dendrite",
        "Saving",
        "Creating previews",
    ],
    "save_traces": [
        "Loading",
        "Extracting",
        "Computing",
        "Saving",
    ],
    "analyze_traces": [
        "Loading",
        "Analyzing",
        "Plotting",
        "Saving",
    ],
}


class ConsoleOutputWidget(QWidget):
    """Widget for displaying console output with progress tracking.

    Provides a scrollable text area for real-time script output with
    auto-scroll, a progress bar, and a clear button.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the console output widget.

        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self._auto_scroll = True
        self._current_steps: List[str] = []
        self._matched_step = 0
        self._captured_stats: dict = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with title and clear button
        header_layout = QHBoxLayout()

        title = QLabel("Console Output")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_output)
        header_layout.addWidget(self.clear_button)

        layout.addLayout(header_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% — %v/%m")
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3e3e3e;
                border-radius: 3px;
                background-color: #2d2d2d;
                text-align: center;
                color: #d4d4d4;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
        """)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Status label for current step
        self.step_label = QLabel()
        self.step_label.setStyleSheet(
            "color: #888; font-size: 11px; padding: 2px 4px;"
        )
        self.step_label.hide()
        layout.addWidget(self.step_label)

        # Text area for output
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)

        # Set monospace font
        mono_font = QFont("Courier New", 10)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self.text_area.setFont(mono_font)

        # Style the text area
        self.text_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                border-radius: 3px;
                padding: 5px;
            }
        """)

        layout.addWidget(self.text_area)

        # Results summary panel (hidden by default)
        self.summary_frame = QWidget()
        self.summary_frame.setStyleSheet("""
            QWidget {
                background-color: #1a2a1a;
                border: 1px solid #4caf50;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        summary_layout = QVBoxLayout(self.summary_frame)
        summary_layout.setContentsMargins(10, 8, 10, 8)

        self.summary_title = QLabel("📊 Results Summary")
        self.summary_title.setStyleSheet(
            "color: #4caf50; font-size: 13px; font-weight: bold; border: none;"
        )
        summary_layout.addWidget(self.summary_title)

        self.summary_text = QLabel()
        self.summary_text.setStyleSheet(
            "color: #d4d4d4; font-size: 12px; border: none;"
        )
        self.summary_text.setWordWrap(True)
        summary_layout.addWidget(self.summary_text)

        # View Plot button
        self.view_plot_button = QPushButton("📊 View Activity Timeline")
        self.view_plot_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        self.view_plot_button.hide()
        summary_layout.addWidget(self.view_plot_button)

        self.summary_frame.hide()
        layout.addWidget(self.summary_frame)

    def start_progress(self, script_name: str) -> None:
        """Start progress tracking for a script.

        Detects the script type and sets up step tracking.

        Args:
            script_name: Name of the script being executed
        """
        self._current_steps = []
        self._matched_step = 0

        # Find matching step list
        for key, steps in MODULE_STEPS.items():
            if key in script_name.lower():
                self._current_steps = steps
                break

        if self._current_steps:
            total = len(self._current_steps)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"0/{total} steps")
            self.progress_bar.show()
            self.step_label.setText("Starting...")
            self.step_label.show()
        else:
            # No known steps - show indeterminate progress
            self.progress_bar.setMaximum(0)  # indeterminate
            self.progress_bar.show()
            self.step_label.setText("Running...")
            self.step_label.show()

    def stop_progress(self, success: bool = True) -> None:
        """Stop progress tracking.

        Args:
            success: Whether the script completed successfully
        """
        if self._current_steps:
            total = len(self._current_steps)
            if success:
                self.progress_bar.setValue(total)
                self.progress_bar.setFormat(f"{total}/{total} steps — Done!")
                self.step_label.setText("✓ Completed")
                self.step_label.setStyleSheet(
                    "color: #4caf50; font-size: 11px; padding: 2px 4px;"
                )
            else:
                self.progress_bar.setFormat("Failed")
                self.step_label.setText("✗ Failed")
                self.step_label.setStyleSheet(
                    "color: #f44336; font-size: 11px; padding: 2px 4px;"
                )
        else:
            self.progress_bar.setMaximum(100)
            if success:
                self.progress_bar.setValue(100)
                self.step_label.setText("✓ Completed")
                self.step_label.setStyleSheet(
                    "color: #4caf50; font-size: 11px; padding: 2px 4px;"
                )
            else:
                self.step_label.setText("✗ Failed")
                self.step_label.setStyleSheet(
                    "color: #f44336; font-size: 11px; padding: 2px 4px;"
                )

        self._current_steps = []
        self._matched_step = 0

    def _check_progress(self, text: str) -> None:
        """Check if output text matches a progress step or contains stats.

        Args:
            text: Output line to check
        """
        # Extract stats from output
        self._extract_stats(text)

        if not self._current_steps:
            return

        # Check if text matches the next expected step (or any later step)
        for i in range(self._matched_step, len(self._current_steps)):
            step = self._current_steps[i]
            if step.lower() in text.lower():
                self._matched_step = i + 1
                total = len(self._current_steps)
                self.progress_bar.setValue(self._matched_step)
                self.progress_bar.setFormat(
                    f"{self._matched_step}/{total} steps"
                )
                # Show a clean step description
                self.step_label.setText(f"⏳ {text.strip('.')}")
                self.step_label.setStyleSheet(
                    "color: #888; font-size: 11px; padding: 2px 4px;"
                )
                break

    def _extract_stats(self, text: str) -> None:
        """Extract statistics from script output.

        Args:
            text: Output line to check for stats
        """
        text_lower = text.lower()

        # M1: "Detected N active frames"
        if "detected" in text_lower and "active frames" in text_lower:
            match = re.search(r'(\d+)\s+active frames', text)
            if match:
                self._captured_stats["active_frames"] = int(match.group(1))

        # M1: "Grouped into N events"
        if "grouped into" in text_lower and "event" in text_lower:
            match = re.search(r'(\d+)\s+events?', text)
            if match:
                self._captured_stats["events"] = int(match.group(1))

        # M1: "Shape: (T, Z, Y, X)"
        if "shape:" in text_lower and "(" in text:
            match = re.search(r'Shape:\s*\((\d+)', text)
            if match:
                self._captured_stats["frames"] = int(match.group(1))

        # M1: "Saved N grouped event crops"
        if "saved" in text_lower and "event crops" in text_lower:
            match = re.search(r'(\d+)\s+grouped event crops', text)
            if match:
                self._captured_stats["event_crops"] = int(match.group(1))

        # M2: masks/dendrites detected
        if "dendrite" in text_lower and ("found" in text_lower or "detected" in text_lower):
            match = re.search(r'(\d+)', text)
            if match:
                self._captured_stats["dendrites"] = int(match.group(1))

        # M4: traces extracted
        if "trace" in text_lower and ("extracted" in text_lower or "saved" in text_lower):
            match = re.search(r'(\d+)', text)
            if match:
                self._captured_stats["traces"] = int(match.group(1))

    def get_captured_stats(self) -> dict:
        """Get statistics captured from script output.

        Returns:
            Dictionary of captured statistics
        """
        return self._captured_stats.copy()

    def show_summary(self, stats: Optional[dict] = None) -> None:
        """Show a results summary panel.

        Args:
            stats: Optional stats dict to display. If None, uses captured stats.
        """
        if stats is None:
            stats = self._captured_stats

        if not stats:
            return

        lines = []
        if "frames" in stats:
            lines.append(f"📹 Total frames: {stats['frames']:,}")
        if "active_frames" in stats:
            lines.append(f"⚡ Active frames: {stats['active_frames']:,}")
        if "events" in stats:
            lines.append(f"🎯 Events detected: {stats['events']}")
        if "event_crops" in stats:
            lines.append(f"📦 Event crops saved: {stats['event_crops']}")
        if "dendrites" in stats:
            lines.append(f"🌿 Dendrites found: {stats['dendrites']}")
        if "traces" in stats:
            lines.append(f"📈 Traces extracted: {stats['traces']}")

        if lines:
            self.summary_text.setText("\n".join(lines))
            self.summary_frame.show()

    def append_output(self, text: str) -> None:
        """Append text to the output area.

        Adds text to the console output, checks for progress updates,
        and auto-scrolls to the bottom if enabled.

        Args:
            text: Text to append
        """
        # Check for progress
        self._check_progress(text)

        # Move cursor to end
        cursor = self.text_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_area.setTextCursor(cursor)

        # Insert text with newline
        self.text_area.insertPlainText(text + "\n")

        # Auto-scroll to bottom if enabled
        if self._auto_scroll:
            self.scroll_to_bottom()

    def scroll_to_bottom(self) -> None:
        """Scroll the text area to the bottom."""
        scrollbar = self.text_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_output(self) -> None:
        """Clear all output from the text area."""
        self.text_area.clear()
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        self.step_label.hide()
        self.step_label.setStyleSheet(
            "color: #888; font-size: 11px; padding: 2px 4px;"
        )
        self.summary_frame.hide()
        self._captured_stats = {}

    def set_auto_scroll(self, enabled: bool) -> None:
        """Enable or disable auto-scrolling.

        Args:
            enabled: True to enable auto-scroll, False to disable
        """
        self._auto_scroll = enabled

    def get_output(self) -> str:
        """Get all text from the output area.

        Returns:
            Complete console output text
        """
        return self.text_area.toPlainText()
