"""
Model layer for MAVCA Pipeline UI.

This module contains the core data models and business logic:
- PipelineConfig: Configuration management (DATE, MOUSE, RUN, base_dir)
- ProgressState: Progress tracking for pipeline execution
- ModuleDefinition: Module metadata and script information
- ScriptExecutor: Script execution and output capture
- StateManager: State persistence and loading
"""

from .pipeline_config import PipelineConfig
from .progress_state import ProgressState
from .state_manager import StateManager
from .module_definition import ModuleDefinition, ScriptInfo
from .script_executor import ScriptExecutor
from .pipeline_modules import (
    MODULES,
    get_module,
    get_all_module_ids,
    get_optional_modules,
    get_interactive_modules,
)

__all__ = [
    "PipelineConfig",
    "ProgressState",
    "ModuleDefinition",
    "ScriptInfo",
    "ScriptExecutor",
    "StateManager",
    "MODULES",
    "get_module",
    "get_all_module_ids",
    "get_optional_modules",
    "get_interactive_modules",
]
