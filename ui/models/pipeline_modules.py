"""Pipeline module configurations for the MAVCA Pipeline UI.

This module defines the complete configuration for all 8 pipeline modules
(M1, M1.5, M2, M2.5, M3, M4, M5, M6) including their scripts, expected outputs,
and metadata.
"""

from pathlib import Path
from typing import Dict

from .module_definition import ModuleDefinition, ScriptInfo


# Complete module configuration for the MAVCA pipeline
MODULES: Dict[str, ModuleDefinition] = {
    "M1": ModuleDefinition(
        id="M1",
        name="Preprocessing & Event Detection",
        description="Detect motion, compute ΔF/F, detect calcium events, extract mini-stacks",
        scripts=[
            ScriptInfo(
                name="find_events_m1.py",
                path=Path("code/Preprocessing-STEP1/find_events_m1.py"),
                description="Main preprocessing and event detection",
                parameters={
                    "CROP_RADIUS": 5,
                    "START_THRESHOLD": 0.5,
                    "END_THRESHOLD": -0.5,
                    "MAX_FRAME_GAP": 2,
                    "Y_CROP": 3
                }
            ),
            ScriptInfo(
                name="ach_ca_plots.py",
                path=Path("code/Preprocessing-STEP1/ach_ca_plots.py"),
                description="Optional: Preview two-channel data",
                optional=True
            )
        ],
        expected_outputs=[
            "preprocessed/raw_clean.tif",
            "preprocessed/event_crops/",
            "preprocessed/excluded_frames.npy",
            "preprocessed/frame_mapping.npy"
        ]
    ),
    
    "M1.5": ModuleDefinition(
        id="M1.5",
        name="Pre-segmentation",
        description="Generate pre-segmentation masks for better initial segmentation",
        scripts=[
            ScriptInfo(
                name="pre_segmentation_m1.5.py",
                path=Path("code/Preprocessing-STEP1/pre_segmentation_m1.5.py"),
                description="Create pre-segmentation masks"
            )
        ],
        expected_outputs=[
            "preprocessed/preseg_masks/"
        ],
        optional=True
    ),
    
    "M2": ModuleDefinition(
        id="M2",
        name="Initial Mask Creation",
        description="Auto-segment dendrites from event crops",
        scripts=[
            ScriptInfo(
                name="detect_masks_m2.py",
                path=Path("code/Masks-STEP2/detect_masks_m2.py"),
                description="Automatic dendrite segmentation",
                parameters={
                    "INTENSITY_PERCENTILE": 95,
                    "MIN_VOL": 100,
                    "MAX_VOL": 10000
                }
            )
        ],
        expected_outputs=[
            "labelmaps/",
            "labelmap_backgrounds/",
            "labelmap_previews/",
            "masks_manifest.csv"
        ]
    ),
    
    "M2.5": ModuleDefinition(
        id="M2.5",
        name="Mask Refinement (Optional)",
        description="Draw trunk guides for improved segmentation",
        scripts=[
            ScriptInfo(
                name="mask_guide_m2.5.py",
                path=Path("code/Masks-STEP2/mask_guide_m2.5.py"),
                description="Interactive guide drawing tool"
            )
        ],
        expected_outputs=[
            "preprocessed/guides/"
        ],
        optional=True
    ),
    
    "M3": ModuleDefinition(
        id="M3",
        name="Mask Curation",
        description="Interactively curate masks with DFF preview",
        scripts=[
            ScriptInfo(
                name="filter_selected_masks_m3.py",
                path=Path("code/Masks-STEP2/filter_selected_masks_m3.py"),
                description="Interactive mask curation in Napari"
            )
        ],
        expected_outputs=[
            "labelmaps_curated_dynamic/"
        ],
        requires_interaction=True
    ),
    
    "M4": ModuleDefinition(
        id="M4",
        name="Trace Extraction",
        description="Extract ΔF/F traces from curated masks",
        scripts=[
            ScriptInfo(
                name="save_traces_m4.py",
                path=Path("code/Traces-STEP3/save_traces_m4.py"),
                description="Extract and save DFF traces"
            )
        ],
        expected_outputs=[
            "traces/dff_traces_curated_bgsub.csv",
            "trace_previews_curated/"
        ]
    ),
    
    "M5": ModuleDefinition(
        id="M5",
        name="Trace Analysis",
        description="Analyze and visualize extracted traces",
        scripts=[
            ScriptInfo(
                name="analyze_traces_m5.py",
                path=Path("code/Traces-STEP3/analyze_traces_m5.py"),
                description="Main trace analysis"
            ),
            ScriptInfo(
                name="depth_analysis_plots.py",
                path=Path("code/Extra/depth_analysis_plots.py"),
                description="Depth-stratified analysis",
                optional=True
            ),
            ScriptInfo(
                name="outside_mask_plot.py",
                path=Path("code/Traces-STEP3/outside_mask_plot.py"),
                description="Inside vs outside mask dynamics",
                optional=True
            )
        ],
        expected_outputs=[
            "traces/dff_traces_curated_bgsub_smooth.csv",
            "traces/depth_analysis_global_ca.png",
            "outside_mask_dynamics/"
        ]
    ),
    
    "M6": ModuleDefinition(
        id="M6",
        name="Visualization",
        description="Create 3D movie visualizations",
        scripts=[
            ScriptInfo(
                name="create_3d_movie_m6.py",
                path=Path("code/Visual-STEP4/create_3d_movie_m6.py"),
                description="Create full 3D movie"
            ),
            ScriptInfo(
                name="create_3d_movie_chunks_m6.py",
                path=Path("code/Visual-STEP4/create_3d_movie_chunks_m6.py"),
                description="Create 3D movie in chunks",
                optional=True
            )
        ],
        expected_outputs=[
            "overlays_curated/"
        ]
    )
}


def get_module(module_id: str) -> ModuleDefinition:
    """Get a module definition by ID.
    
    Args:
        module_id: The module identifier (e.g., "M1", "M1.5")
        
    Returns:
        The ModuleDefinition for the specified module
        
    Raises:
        KeyError: If the module_id is not found
    """
    return MODULES[module_id]


def get_all_module_ids() -> list[str]:
    """Get a list of all module IDs in order.
    
    Returns:
        List of module IDs: ["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]
    """
    return ["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]


def get_optional_modules() -> list[str]:
    """Get a list of optional module IDs.
    
    Returns:
        List of optional module IDs: ["M1.5", "M2.5"]
    """
    return [module_id for module_id, module in MODULES.items() if module.optional]


def get_interactive_modules() -> list[str]:
    """Get a list of module IDs that require user interaction.
    
    Returns:
        List of interactive module IDs: ["M3"]
    """
    return [module_id for module_id, module in MODULES.items() if module.requires_interaction]
