"""
Extension modules for the MAVCA Pipeline UI.

This package provides extension points for future features including:
- M6 3D visualization
- Behavioral data integration
- Custom analysis modules
"""

from .m6_visualization import (
    M6VisualizationWidget,
    M6VisualizationExtension,
    get_m6_extension,
    show_m6_visualization_info
)
from .behavioral_analysis import (
    BehavioralAnalysisWidget,
    BehavioralAnalysisExtension,
    BehavioralDataType,
    get_behavioral_extension,
    show_behavioral_analysis_info
)

__all__ = [
    'M6VisualizationWidget',
    'M6VisualizationExtension',
    'get_m6_extension',
    'show_m6_visualization_info',
    'BehavioralAnalysisWidget',
    'BehavioralAnalysisExtension',
    'BehavioralDataType',
    'get_behavioral_extension',
    'show_behavioral_analysis_info'
]
