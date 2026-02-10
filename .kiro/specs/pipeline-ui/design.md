# Design Document: MAVCA Pipeline UI

## Overview

The MAVCA Pipeline UI is a desktop application that provides a graphical interface for the Mask-Assisted Volumetric Calcium Analysis pipeline. The application will be built using Python with a modern GUI framework to ensure seamless integration with the existing Python codebase while providing an intuitive user experience for researchers.

### Key Design Goals

1. **Minimal Code Changes**: Integrate with existing Python scripts without requiring modifications
2. **Progressive Disclosure**: Show complexity only when needed, keeping the interface clean
3. **Visual Feedback**: Provide clear indication of progress, status, and next steps
4. **Extensibility**: Design for future additions (M6 visualization, behavioral data)
5. **Reliability**: Robust error handling and state persistence

### Technology Stack

**GUI Framework**: PyQt6 (or PySide6)
- Native look and feel on macOS
- Rich widget library for complex interfaces
- Excellent support for embedding matplotlib plots
- Strong integration with Python scientific stack
- Active community and documentation

**Alternative Considered**: Tkinter
- Rejected due to limited styling capabilities and less modern appearance
- Insufficient support for complex visualizations needed for M3 curation

**State Management**: JSON-based persistence
- Simple, human-readable format
- Easy to debug and manually edit if needed
- Stores progress state per dataset (DATE/MOUSE/RUN combination)

**Process Execution**: Python subprocess module
- Captures stdout/stderr in real-time
- Allows script termination
- Maintains isolation between UI and analysis scripts

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Main Window                          │
│  ┌──────────────┐  ┌──────────────────────────────┐   │
│  │  Parameter   │  │     Module View              │   │
│  │  Panel       │  │  ┌────────────────────────┐  │   │
│  │              │  │  │  Module Details        │  │   │
│  │  DATE        │  │  │  - Scripts             │  │   │
│  │  MOUSE       │  │  │  - Expected Outputs    │  │   │
│  │  RUN         │  │  │  - Execution Controls  │  │   │
│  │              │  │  └────────────────────────┘  │   │
│  │  [Base Dir]  │  │                              │   │
│  └──────────────┘  └──────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │          Module Navigation Bar                   │  │
│  │  [M1] [M1.5] [M2] [M2.5] [M3] [M4] [M5] [M6]   │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │          Console Output Panel                    │  │
│  │  (Real-time script output)                       │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              M3 Curation Window (Separate)              │
│  ┌──────────────────────────────────────────────────┐  │
│  │         3D Mask Visualization                     │  │
│  │         (Napari or custom viewer)                 │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         DFF Trace Preview                         │  │
│  │         (Matplotlib embedded plot)                │  │
│  └──────────────────────────────────────────────────┘  │
│  [Previous] [Next] [Keep] [Delete] [Merge] [Save]     │
└─────────────────────────────────────────────────────────┘
```

### Component Architecture

The application follows a Model-View-Controller (MVC) pattern:

**Model Layer**:
- `PipelineConfig`: Manages DATE, MOUSE, RUN, base directory
- `ProgressState`: Tracks module completion status per dataset
- `ModuleDefinition`: Defines module metadata, scripts, expected outputs
- `ScriptExecutor`: Handles Python script execution and output capture

**View Layer**:
- `MainWindow`: Primary application window
- `ParameterPanel`: Configuration inputs
- `ModuleNavigationBar`: Module selection tabs
- `ModuleDetailView`: Displays module information and controls
- `ConsoleOutputWidget`: Real-time script output display
- `CurationWindow`: M3 interactive curation interface
- `PlotViewerWidget`: Displays analysis plots from M4/M5

**Controller Layer**:
- `PipelineController`: Coordinates between model and view
- `ScriptRunner`: Manages script execution lifecycle
- `StateManager`: Handles persistence and state transitions

## Components and Interfaces

### 1. PipelineConfig

Manages the common parameters and base directory configuration.

```python
class PipelineConfig:
    """Configuration for pipeline execution"""
    
    def __init__(self):
        self.date: str = ""
        self.mouse: str = ""
        self.run: str = ""
        self.base_dir: Path = Path.home() / "Desktop" / "Boston_University" / "Devor_Lab" / "apical-dendrites-2025" / "scape-data"
    
    def validate_date(self, date_str: str) -> bool:
        """Validate DATE format (YYYY-MM-DD)"""
        pass
    
    def validate_identifier(self, identifier: str) -> bool:
        """Validate MOUSE or RUN (alphanumeric, underscore, hyphen)"""
        pass
    
    def get_data_path(self) -> Path:
        """Construct data/DATE/MOUSE/RUN path"""
        pass
    
    def data_path_exists(self) -> bool:
        """Check if data directory exists"""
        pass
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        pass
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PipelineConfig':
        """Deserialize from dictionary"""
        pass
```

### 2. ProgressState

Tracks which modules have been completed for a specific dataset.

```python
class ProgressState:
    """Progress tracking for a specific dataset"""
    
    def __init__(self, dataset_key: str):
        self.dataset_key: str = dataset_key  # "DATE_MOUSE_RUN"
        self.completed_modules: Set[str] = set()
        self.m2_guided: bool = False  # Whether M2 was run with guides
        self.last_updated: datetime = datetime.now()
    
    def mark_complete(self, module_id: str) -> None:
        """Mark a module as completed"""
        pass
    
    def is_complete(self, module_id: str) -> bool:
        """Check if a module is completed"""
        pass
    
    def reset(self) -> None:
        """Reset all progress"""
        pass
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        pass
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProgressState':
        """Deserialize from dictionary"""
        pass
```

### 3. ModuleDefinition

Defines the metadata and configuration for each pipeline module.

```python
@dataclass
class ScriptInfo:
    """Information about a script within a module"""
    name: str
    path: Path
    description: str
    optional: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModuleDefinition:
    """Definition of a pipeline module"""
    id: str  # "M1", "M1.5", etc.
    name: str
    description: str
    scripts: List[ScriptInfo]
    expected_outputs: List[str]  # Relative paths from data/DATE/MOUSE/RUN/
    optional: bool = False
    requires_interaction: bool = False  # True for M3
    
    def get_script_by_name(self, name: str) -> Optional[ScriptInfo]:
        """Find a script by name"""
        pass
```

### 4. ScriptExecutor

Handles execution of Python scripts with real-time output capture.

```python
class ScriptExecutor:
    """Executes Python scripts and captures output"""
    
    def __init__(self, venv_path: Path):
        self.venv_path: Path = venv_path
        self.process: Optional[subprocess.Popen] = None
        self.output_callback: Optional[Callable[[str], None]] = None
    
    def execute_script(
        self,
        script_path: Path,
        config: PipelineConfig,
        output_callback: Callable[[str], None]
    ) -> int:
        """
        Execute a script with given configuration.
        Returns exit code.
        """
        pass
    
    def terminate(self) -> None:
        """Terminate the running process"""
        pass
    
    def is_running(self) -> bool:
        """Check if a process is currently running"""
        pass
    
    def _inject_parameters(
        self,
        script_path: Path,
        config: PipelineConfig
    ) -> Path:
        """
        Create a temporary modified script with injected parameters.
        This allows parameter override without modifying original scripts.
        """
        pass
```

### 5. StateManager

Manages persistence of configuration and progress state.

```python
class StateManager:
    """Manages application state persistence"""
    
    def __init__(self, state_dir: Path):
        self.state_dir: Path = state_dir
        self.config_file: Path = state_dir / "config.json"
        self.progress_file: Path = state_dir / "progress.json"
    
    def save_config(self, config: PipelineConfig) -> None:
        """Save configuration to disk"""
        pass
    
    def load_config(self) -> Optional[PipelineConfig]:
        """Load configuration from disk"""
        pass
    
    def save_progress(self, progress: ProgressState) -> None:
        """Save progress state to disk"""
        pass
    
    def load_progress(self, dataset_key: str) -> ProgressState:
        """Load progress state for a dataset"""
        pass
    
    def list_datasets(self) -> List[str]:
        """List all datasets with saved progress"""
        pass
```

### 6. MainWindow (View)

The primary application window.

```python
class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, controller: PipelineController):
        super().__init__()
        self.controller = controller
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Initialize UI components"""
        pass
    
    def update_module_view(self, module: ModuleDefinition) -> None:
        """Update the module detail view"""
        pass
    
    def update_progress_indicators(self, progress: ProgressState) -> None:
        """Update visual progress indicators"""
        pass
    
    def append_console_output(self, text: str) -> None:
        """Append text to console output"""
        pass
    
    def show_error(self, message: str) -> None:
        """Display error dialog"""
        pass
    
    def show_warning(self, message: str) -> None:
        """Display warning dialog"""
        pass
```

### 7. CurationWindow (View)

Specialized window for M3 mask curation.

```python
class CurationWindow(QMainWindow):
    """Interactive mask curation interface for M3"""
    
    def __init__(self, data_path: Path, controller: PipelineController):
        super().__init__()
        self.data_path = data_path
        self.controller = controller
        self.current_mask_index = 0
        self.masks: List[MaskData] = []
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Initialize curation UI"""
        pass
    
    def load_masks(self) -> None:
        """Load all masks from labelmaps directory"""
        pass
    
    def display_mask(self, index: int) -> None:
        """Display mask at given index"""
        pass
    
    def compute_dff_preview(self, mask: MaskData) -> np.ndarray:
        """Compute DFF trace for current mask"""
        pass
    
    def update_trace_plot(self, trace: np.ndarray) -> None:
        """Update the DFF trace plot"""
        pass
    
    def on_next(self) -> None:
        """Navigate to next mask"""
        pass
    
    def on_previous(self) -> None:
        """Navigate to previous mask"""
        pass
    
    def on_keep(self) -> None:
        """Mark current mask as kept"""
        pass
    
    def on_delete(self) -> None:
        """Mark current mask for deletion"""
        pass
    
    def on_merge(self) -> None:
        """Initiate merge operation"""
        pass
    
    def save_curation_results(self) -> None:
        """Save curated masks to disk"""
        pass
```

### 8. PipelineController

Coordinates application logic and state management.

```python
class PipelineController:
    """Main controller for pipeline operations"""
    
    def __init__(self):
        self.config = PipelineConfig()
        self.state_manager = StateManager(Path.home() / ".mavca_ui")
        self.executor = ScriptExecutor(self._find_venv())
        self.modules = self._load_module_definitions()
        self.current_progress: Optional[ProgressState] = None
    
    def initialize(self) -> None:
        """Initialize controller and load saved state"""
        pass
    
    def update_config(self, date: str, mouse: str, run: str) -> bool:
        """Update configuration and load corresponding progress"""
        pass
    
    def execute_module(self, module_id: str, output_callback: Callable[[str], None]) -> None:
        """Execute all scripts for a module"""
        pass
    
    def verify_outputs(self, module_id: str) -> Dict[str, bool]:
        """Check which expected outputs exist"""
        pass
    
    def launch_curation(self) -> None:
        """Launch M3 curation window"""
        pass
    
    def open_output_directory(self, module_id: str) -> None:
        """Open module output directory in file browser"""
        pass
    
    def _load_module_definitions(self) -> Dict[str, ModuleDefinition]:
        """Load module definitions from configuration"""
        pass
    
    def _find_venv(self) -> Path:
        """Locate the project's virtual environment"""
        pass
```

## Data Models

### Module Configuration

Modules are defined in a configuration structure that can be easily extended:

```python
MODULES = {
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
```

### Progress State Storage

Progress is stored per dataset in JSON format:

```json
{
  "2025-12-25_rAi162_phpeb_run1": {
    "completed_modules": ["M1", "M1.5", "M2", "M2.5", "M2", "M3"],
    "m2_guided": true,
    "last_updated": "2025-01-15T14:30:00"
  },
  "2025-12-26_organoid_run4-crop": {
    "completed_modules": ["M1", "M1.5", "M2"],
    "m2_guided": false,
    "last_updated": "2025-01-16T09:15:00"
  }
}
```

### Configuration Storage

Application configuration is stored in JSON:

```json
{
  "date": "2025-12-25",
  "mouse": "rAi162_phpeb",
  "run": "run1",
  "base_dir": "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data",
  "window_geometry": {
    "x": 100,
    "y": 100,
    "width": 1200,
    "height": 800
  }
}
```

## Error Handling

### Error Categories

1. **Configuration Errors**
   - Invalid DATE format
   - Invalid MOUSE/RUN identifiers
   - Missing base directory
   - Missing virtual environment

2. **Execution Errors**
   - Script not found
   - Python interpreter not found
   - Script execution failure (non-zero exit code)
   - Script timeout

3. **Data Errors**
   - Missing input files
   - Missing expected outputs after execution
   - Corrupted data files

4. **State Errors**
   - Corrupted state files
   - Permission errors when saving state

### Error Handling Strategy

**Configuration Errors**: Validate immediately on input, show inline error messages, prevent execution until resolved.

**Execution Errors**: Display full error output in console, maintain previous progress state, offer retry option.

**Data Errors**: Show warnings before execution if inputs missing, show warnings after execution if outputs missing, provide option to open directory for manual inspection.

**State Errors**: Fall back to default state if corrupted, log errors for debugging, continue operation with in-memory state.

### User-Friendly Error Messages

```python
ERROR_MESSAGES = {
    "invalid_date": "Date must be in YYYY-MM-DD format (e.g., 2025-12-25)",
    "invalid_identifier": "Must contain only letters, numbers, underscores, and hyphens",
    "missing_data_dir": "Data directory does not exist: {path}\nPlease check your DATE, MOUSE, and RUN values.",
    "missing_venv": "Could not find Python virtual environment.\nPlease ensure .venv311 or .venv exists in the project directory.",
    "script_failed": "Script failed with error code {code}.\nCheck the console output for details.",
    "missing_inputs": "Required input files are missing:\n{files}\nPlease run previous modules first.",
    "missing_outputs": "Expected outputs were not created:\n{files}\nThe script may have failed. Check the console output."
}
```

## Testing Strategy

The testing strategy combines unit tests for individual components and integration tests for end-to-end workflows. Property-based testing will be used to verify universal correctness properties across all inputs.

### Unit Testing

Unit tests will focus on:
- Configuration validation logic
- Path construction and validation
- State serialization/deserialization
- Module definition parsing
- Error message generation

### Integration Testing

Integration tests will verify:
- Complete workflow from M1 through M6
- State persistence across application restarts
- Script execution with parameter injection
- Output verification logic
- M2/M2.5 iteration workflow

### Property-Based Testing

Property-based tests will verify universal properties using a Python property testing library (Hypothesis). Each test will run a minimum of 100 iterations with randomized inputs.

**Property Testing Library**: Hypothesis (Python)
- Mature, well-documented library
- Excellent integration with pytest
- Rich set of strategies for generating test data
- Shrinking capability to find minimal failing examples

**Test Configuration**: Each property test will:
- Run minimum 100 iterations
- Include a comment tag: `# Feature: pipeline-ui, Property N: <property text>`
- Reference the corresponding design document property number


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Module Information Completeness

*For any* module definition, when rendered in the UI, the display should include the module number, name, and purpose.

**Validates: Requirements 1.2**

### Property 2: Optional Module Indication

*For any* module with the optional flag set, the UI should visually indicate that the module is optional.

**Validates: Requirements 1.3**

### Property 3: Current Module Visual Indication

*For any* module selection, the UI state should reflect that module as the currently active module.

**Validates: Requirements 1.4**

### Property 4: Script List Completeness

*For any* module with N scripts, the UI should display all N scripts.

**Validates: Requirements 1.5**

### Property 5: Date Format Validation

*For any* string input, the date validation should accept it if and only if it matches the YYYY-MM-DD format with valid month (01-12) and day values.

**Validates: Requirements 2.2**

### Property 6: Identifier Validation

*For any* string input, the identifier validation (for MOUSE and RUN) should accept it if and only if it contains only alphanumeric characters, underscores, and hyphens.

**Validates: Requirements 2.3, 2.4**

### Property 7: Configuration Persistence Round-Trip

*For any* valid configuration (DATE, MOUSE, RUN, base_dir), saving then loading the configuration should produce an equivalent configuration.

**Validates: Requirements 2.5**

### Property 8: Parameter Change Indication

*For any* parameter change, the UI should update to indicate that the new values will be used on next execution.

**Validates: Requirements 2.6**

### Property 9: Module Execution Controls Availability

*For any* module, the UI should provide execution controls for that module.

**Validates: Requirements 3.1**

### Property 10: Execution Exclusivity

*For any* running script execution, attempting to start another script execution should be prevented.

**Validates: Requirements 3.3**

### Property 11: Successful Execution Updates Progress

*For any* successful script execution, the progress state should be updated to mark the corresponding module as completed.

**Validates: Requirements 3.4, 6.2**

### Property 12: Failed Execution Preserves Progress

*For any* failed script execution, the progress state should remain unchanged from its pre-execution state.

**Validates: Requirements 3.5**

### Property 13: Parameter Injection

*For any* script execution, the injected DATE, MOUSE, and RUN parameters should match the currently configured values.

**Validates: Requirements 3.7**

### Property 14: Guided Labelmap Tracking

*For any* progress state, the m2_guided flag should correctly reflect whether M2 was last run with guides enabled.

**Validates: Requirements 4.6, 6.7**

### Property 15: Mask Selection Displays Trace

*For any* mask selection in the M3 curation interface, a DFF trace preview should be computed and displayed.

**Validates: Requirements 5.3**

### Property 16: Mask Navigation Completeness

*For any* dataset with N masks, the navigation controls should allow reaching all N masks through sequential next/previous operations.

**Validates: Requirements 5.5**

### Property 17: Curation Operation Updates State

*For any* curation operation (Keep, Delete, Merge), the mask state should be immediately updated to reflect the operation.

**Validates: Requirements 5.6**

### Property 18: Curation State Persistence Round-Trip

*For any* curation session with operations performed, exiting and resuming should preserve all curation state (which masks were kept, deleted, or merged).

**Validates: Requirements 5.8**

### Property 19: Progress State Tracking

*For any* set of completed modules, the progress state should accurately reflect which modules are completed.

**Validates: Requirements 6.1**

### Property 20: Completion Status Visual Distinction

*For any* module, the UI should visually distinguish whether it is completed or incomplete based on the progress state.

**Validates: Requirements 6.3**

### Property 21: Progress Persistence Round-Trip

*For any* progress state for a dataset, saving then loading should produce an equivalent progress state.

**Validates: Requirements 6.4**

### Property 22: Progress Reset

*For any* progress state with completed modules, resetting should clear all completions.

**Validates: Requirements 6.5**

### Property 23: Dataset-Specific Progress Loading

*For any* dataset key (DATE_MOUSE_RUN), changing to that dataset should load the correct progress state for that specific dataset.

**Validates: Requirements 6.6**

### Property 24: Expected Outputs Display

*For any* module, the UI should display all expected output paths for that module.

**Validates: Requirements 7.1**

### Property 25: Output Existence Verification

*For any* completed module, the UI should check and indicate which expected outputs exist on the filesystem.

**Validates: Requirements 7.2**

### Property 26: Missing Output Warning

*For any* completed module with missing expected outputs, the UI should display a visual warning.

**Validates: Requirements 7.3**

### Property 27: Output Path Construction

*For any* valid DATE, MOUSE, and RUN configuration, the constructed output paths should follow the pattern data/DATE/MOUSE/RUN/.

**Validates: Requirements 7.5, 11.1, 11.2**

### Property 28: Module Navigation Display

*For any* module navigation, the UI should display that module's details, scripts, and expected outputs.

**Validates: Requirements 8.2**

### Property 29: Unrestricted Module Navigation

*For any* module regardless of completion status, navigation to that module should be possible.

**Validates: Requirements 8.3**

### Property 30: Navigation Preserves Configuration

*For any* sequence of module navigations, the parameter configuration should remain unchanged.

**Validates: Requirements 8.5**

### Property 31: macOS Path Handling

*For any* valid macOS file path (including paths with spaces, special characters, and Unicode), the UI should correctly handle and display the path.

**Validates: Requirements 9.5**

### Property 32: Error Message Display

*For any* error condition, the UI should display an appropriate user-friendly error message.

**Validates: Requirements 10.3**

### Property 33: Window Geometry Persistence Round-Trip

*For any* window geometry (position and size), saving then loading should produce equivalent geometry values.

**Validates: Requirements 10.7**

### Property 34: Data Directory Existence Check

*For any* script execution attempt, the UI should verify that the base data directory exists before execution.

**Validates: Requirements 11.3**

### Property 35: Missing Directory Warning

*For any* non-existent data directory, the UI should display a warning before allowing script execution.

**Validates: Requirements 11.4**

### Property 36: Script Parameters Display

*For any* script with configuration parameters, the UI should display all parameter names, values, and descriptions.

**Validates: Requirements 12.1, 12.2**

### Property 37: Parameter Categorization

*For any* parameter, the UI should correctly categorize it as either common (DATE, MOUSE, RUN) or script-specific.

**Validates: Requirements 12.3**


## Testing Strategy

### Dual Testing Approach

The testing strategy employs both unit tests and property-based tests as complementary approaches:

**Unit Tests**: Focus on specific examples, edge cases, and integration points
- Verify specific UI elements exist (buttons, input fields, navigation controls)
- Test specific module workflows (M2/M2.5 iteration, M3 curation launch)
- Test error conditions with known inputs
- Test integration between components
- Verify correct Python interpreter is used
- Test keyboard shortcuts functionality

**Property-Based Tests**: Focus on universal properties across all inputs
- Configuration validation across all possible inputs
- State persistence round-trips with randomized data
- Path construction with varied parameters
- UI state consistency across operations
- Progress tracking across module sequences

### Property-Based Testing Configuration

**Library**: Hypothesis for Python
- Mature, well-documented property testing framework
- Excellent pytest integration
- Rich strategies for generating test data
- Automatic shrinking to find minimal failing examples

**Test Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with: `# Feature: pipeline-ui, Property N: <property text>`
- Each test references its design document property number
- Tests use Hypothesis strategies to generate:
  - Random date strings (valid and invalid formats)
  - Random identifier strings (valid and invalid characters)
  - Random module configurations
  - Random progress states
  - Random file paths

**Example Property Test Structure**:

```python
from hypothesis import given, strategies as st
import pytest

# Feature: pipeline-ui, Property 5: Date Format Validation
@given(st.text())
def test_date_validation_property(date_string):
    """
    For any string input, date validation should accept it
    if and only if it matches YYYY-MM-DD format with valid values.
    
    Validates: Requirements 2.2
    """
    config = PipelineConfig()
    is_valid = config.validate_date(date_string)
    
    # Check if string matches expected format
    expected_valid = _is_valid_date_format(date_string)
    
    assert is_valid == expected_valid

# Feature: pipeline-ui, Property 7: Configuration Persistence Round-Trip
@given(
    st.text(min_size=10, max_size=10).filter(lambda s: _is_valid_date_format(s)),
    st.text(min_size=1, max_size=50).filter(lambda s: _is_valid_identifier(s)),
    st.text(min_size=1, max_size=50).filter(lambda s: _is_valid_identifier(s))
)
def test_config_persistence_roundtrip(date, mouse, run):
    """
    For any valid configuration, saving then loading should
    produce an equivalent configuration.
    
    Validates: Requirements 2.5
    """
    config = PipelineConfig()
    config.date = date
    config.mouse = mouse
    config.run = run
    
    state_manager = StateManager(tmp_path)
    state_manager.save_config(config)
    loaded_config = state_manager.load_config()
    
    assert loaded_config.date == config.date
    assert loaded_config.mouse == config.mouse
    assert loaded_config.run == config.run
```

### Unit Test Coverage

Unit tests will cover:

1. **UI Element Existence**:
   - All 8 modules displayed in navigation bar (Req 1.1)
   - DATE, MOUSE, RUN input fields present (Req 2.1)
   - Execution controls available (Req 3.1)
   - Stop button available (Req 3.6)
   - M3 curation buttons present (Req 5.2)
   - Navigation controls present (Req 8.1)
   - Overview button present (Req 8.4)
   - Directory open button present (Req 7.4)
   - Base directory configuration available (Req 11.5)
   - Source code viewer available (Req 12.4)

2. **Specific Workflows**:
   - M2 completion shows M3/M2.5 options (Req 4.1)
   - M2.5 selection shows guide drawing notice (Req 4.2)
   - M2.5 completion shows M2 re-run option (Req 4.3)
   - M2 re-run after M2.5 shows guided notice (Req 4.4)
   - Guided/non-guided comparison available (Req 4.5)
   - M3 launch displays mask and controls (Req 5.1)
   - Curation completion saves and updates progress (Req 5.7)
   - M4/M5 completion displays plots (Req 13.1)

3. **Integration Points**:
   - Script output captured and displayed (Req 3.2)
   - Correct Python interpreter used (Req 9.2)
   - Tooltips present for parameters (Req 10.2)
   - Keyboard shortcuts functional (Req 10.6)
   - Plot viewing functionality (Req 13.2, 13.3, 13.4)
   - Plot zoom/pan functionality (Req 13.5)
   - M6 placeholder support (Req 14.2)
   - Behavioral analysis placeholder (Req 14.3)

### Integration Testing

Integration tests will verify end-to-end workflows:

1. **Complete Pipeline Execution**:
   - Configure parameters → Execute M1 → Verify outputs → Execute M1.5 → ... → Execute M6
   - Verify progress tracking throughout
   - Verify state persistence across simulated restarts

2. **M2/M2.5 Iteration**:
   - Execute M2 → Execute M2.5 → Re-execute M2 with guides
   - Verify guided flag tracking
   - Verify labelmap comparison

3. **Multi-Dataset Workflow**:
   - Configure dataset A → Execute modules → Switch to dataset B → Execute modules
   - Verify correct progress loading for each dataset
   - Switch back to dataset A → Verify progress preserved

4. **Error Recovery**:
   - Execute script with missing inputs → Verify error display → Fix inputs → Retry
   - Execute script that fails → Verify progress unchanged → Retry

### Test Data

Tests will use:
- Mock Python scripts that produce predictable output
- Temporary directories for state persistence
- Sample module definitions
- Generated test data (via Hypothesis for property tests)
- Fixture data for integration tests

### Continuous Testing

- All tests run on every commit
- Property tests run with 100 iterations in CI
- Integration tests run with sample datasets
- UI tests run in headless mode where possible
