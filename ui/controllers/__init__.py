"""
Controller layer for MAVCA Pipeline UI.

This module contains the application logic and coordination:
- PipelineController: Main controller coordinating model and view
- ScriptRunner: Script execution lifecycle management
"""

from .pipeline_controller import PipelineController

__all__ = [
    "PipelineController",
]
