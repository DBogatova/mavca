"""Module definition models for the MAVCA Pipeline UI.

This module defines the data structures for pipeline modules and their scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ScriptInfo:
    """Information about a script within a module.
    
    Attributes:
        name: The script filename (e.g., "find_events_m1.py")
        path: Path to the script file relative to project root
        description: Human-readable description of what the script does
        optional: Whether this script is optional (default: False)
        parameters: Dictionary of script-specific configuration parameters
    """
    name: str
    path: Path
    description: str
    optional: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleDefinition:
    """Definition of a pipeline module.
    
    Attributes:
        id: Module identifier (e.g., "M1", "M1.5", "M2")
        name: Human-readable module name
        description: Description of what the module does
        scripts: List of scripts belonging to this module
        expected_outputs: List of expected output paths relative to data/DATE/MOUSE/RUN/
        optional: Whether this module is optional (default: False)
        requires_interaction: Whether this module requires user interaction (default: False)
    """
    id: str
    name: str
    description: str
    scripts: List[ScriptInfo]
    expected_outputs: List[str]
    optional: bool = False
    requires_interaction: bool = False
    
    def get_script_by_name(self, name: str) -> Optional[ScriptInfo]:
        """Find a script by name.
        
        Args:
            name: The script filename to search for
            
        Returns:
            The ScriptInfo object if found, None otherwise
        """
        for script in self.scripts:
            if script.name == name:
                return script
        return None
