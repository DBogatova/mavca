# MAVCA Pipeline UI

A graphical user interface for the Mask-Assisted Volumetric Calcium Analysis (MAVCA) pipeline.

## Overview

The MAVCA Pipeline UI provides researchers with an intuitive desktop application for processing 4D calcium imaging data through a sequential analysis pipeline. The interface simplifies complex workflows, allowing researchers to focus on their science rather than command-line operations.

### Key Features

- **Parameter Configuration**: Set DATE, MOUSE, and RUN parameters in one place
- **Sequential Module Execution**: Run pipeline modules M1 through M6 in order
- **Progress Tracking**: Track completion status across multiple datasets
- **Interactive Curation**: Review and edit 3D masks with real-time DFF trace previews (M3)
- **Output Verification**: Automatically check for expected output files
- **Real-time Feedback**: View script output as it executes
- **Multi-Dataset Support**: Switch between datasets while preserving progress

### Pipeline Modules

The MAVCA pipeline consists of 8 sequential modules:

1. **M1**: Preprocessing & Event Detection - Detect motion, compute ΔF/F, detect calcium events
2. **M1.5**: Pre-segmentation (Optional) - Generate pre-segmentation masks
3. **M2**: Initial Mask Creation - Auto-segment dendrites from event crops
4. **M2.5**: Mask Refinement (Optional) - Draw trunk guides for improved segmentation
5. **M3**: Mask Curation - Interactively review and edit 3D masks
6. **M4**: Trace Extraction - Extract ΔF/F traces from curated masks
7. **M5**: Trace Analysis - Analyze and visualize extracted traces
8. **M6**: Visualization - Create 3D movie visualizations

## Installation

### Prerequisites

- Python 3.10 or higher
- Virtual environment (.venv311 or .venv) in project root
- macOS (primary support), Linux or Windows (experimental)

### Install Dependencies

1. Activate your virtual environment:
```bash
source .venv311/bin/activate  # or .venv/bin/activate
```

2. Install UI dependencies:
```bash
pip install -r requirements_ui.txt
```

3. Verify installation:
```bash
python -c "import PyQt6; print('PyQt6 installed successfully')"
```

## Quick Start

### Launching the Application

From the project root directory:

```bash
python main.py
```

Or with debug logging:

```bash
python main.py --debug
```

### First-Time Setup

1. **Configure Parameters**: In the left panel, enter your dataset parameters:
   - **DATE**: Format YYYY-MM-DD (e.g., 2025-12-25)
   - **MOUSE**: Mouse identifier (e.g., rAi162_phpeb)
   - **RUN**: Run identifier (e.g., run1)

2. **Verify Data Directory**: Click "Browse" to confirm your base data directory location. The application will construct the full path as: `base_dir/data/DATE/MOUSE/RUN/`

3. **Start with M1**: Click on the M1 module in the navigation bar to begin

### Basic Workflow

1. **Select a Module**: Click on a module button (M1, M1.5, M2, etc.) in the navigation bar
2. **Review Module Details**: The center panel shows:
   - Module description and purpose
   - List of scripts that will be executed
   - Expected output files
   - Script parameters
3. **Execute Module**: Click the "Execute Module" button
4. **Monitor Progress**: Watch real-time output in the console panel
5. **Verify Outputs**: After completion, check that expected outputs exist (✓ or ✗ indicators)
6. **Proceed to Next Module**: Move to the next module in sequence

## User Guide

### Interface Layout

The main window is divided into four main areas:

```
┌─────────────────────────────────────────────────────────┐
│  [M1] [M1.5] [M2] [M2.5] [M3] [M4] [M5] [M6]           │  ← Navigation Bar
├──────────────┬──────────────────────────────────────────┤
│  Parameters  │  Module Details                          │
│              │  - Description                           │
│  DATE        │  - Scripts                               │
│  MOUSE       │  - Expected Outputs                      │
│  RUN         │  - [Execute Module]                      │
│              │                                          │
│  [Browse]    │                                          │
├──────────────┴──────────────────────────────────────────┤
│  Console Output                                         │
│  (Real-time script output)                              │
└─────────────────────────────────────────────────────────┘
```

#### Navigation Bar (Top)
- Shows all 8 pipeline modules
- **Green**: Completed modules
- **Blue**: Current module
- **Gray**: Incomplete modules
- **Italic**: Optional modules (M1.5, M2.5)

#### Parameter Panel (Left)
- **DATE**: Dataset date in YYYY-MM-DD format
- **MOUSE**: Mouse identifier
- **RUN**: Run identifier
- **Base Directory**: Root directory for data files
- **Apply**: Save parameter changes

#### Module Detail View (Center)
- Module name and description
- List of scripts to be executed
- Expected output files with existence indicators
- Script parameters and their values
- Execute and Open Directory buttons

#### Console Output (Bottom)
- Real-time output from executing scripts
- Error messages and warnings
- Execution status updates

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+1` through `Ctrl+8` | Navigate to modules M1-M6 |
| `Ctrl+N` | New dataset configuration |
| `Ctrl+Q` | Quit application |
| `Ctrl+Shift+C` | Toggle console visibility |
| `Ctrl+Shift+P` | Toggle plot viewer visibility |
| `Ctrl+L` | Clear console output |
| `F1` | Open documentation |

### Parameter Configuration

#### DATE Format
- Must be in YYYY-MM-DD format
- Example: `2025-12-25`
- Validation: Year 1900-2100, valid month (01-12), valid day

#### MOUSE and RUN Identifiers
- Can contain: letters, numbers, underscores, hyphens
- Examples: `rAi162_phpeb`, `organoid`, `run1`, `run4-crop`
- No spaces or special characters allowed

#### Base Directory
- Default: `~/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data`
- Click "Browse" to change location
- Data path constructed as: `base_dir/data/DATE/MOUSE/RUN/`

### Module Execution

#### Before Executing
1. Ensure parameters are configured correctly
2. Verify data directory exists (warning shown if missing)
3. Check that previous modules are complete (for sequential modules)
4. Review expected outputs to understand what will be created

#### During Execution
- Console shows real-time output
- Execute button is disabled
- Status bar shows current operation
- Cannot start another module while one is running

#### After Execution
- Success: Module marked complete (green in navigation bar)
- Failure: Error message shown, progress unchanged
- Output verification: Check marks (✓) show which outputs exist
- Missing outputs: Warning indicators (✗) shown

#### Stopping Execution
- Currently running scripts cannot be stopped mid-execution
- Close the application to terminate (not recommended)
- Future versions will support graceful termination

### M2/M2.5 Iteration Workflow

The M2/M2.5 workflow allows iterative refinement of mask segmentation:

#### Initial Segmentation (M2)
1. Execute M2 to create initial automatic segmentation
2. Review the generated labelmaps
3. Decide: Proceed to M3 or refine with M2.5?

#### Optional Refinement (M2.5)
1. Execute M2.5 to launch the guide drawing tool
2. Manually draw trunk guides on dendrites that need better segmentation
3. Save guides and exit the tool

#### Re-run with Guides
1. Execute M2 again (it will detect M2.5 guides)
2. M2 creates new "guided" labelmaps using your manual guides
3. Compare guided vs non-guided results
4. Proceed to M3 with the better labelmap set

#### Tips
- M2.5 is optional but recommended for complex datasets
- You can iterate M2→M2.5→M2 multiple times
- Guided labelmaps typically produce better segmentation
- The UI tracks whether your current M2 results are guided or non-guided

### M3 Curation Workflow

M3 provides an interactive interface for reviewing and editing 3D masks:

#### Launching M3
1. Complete M2 (and optionally M2.5)
2. Navigate to M3 module
3. Click "Execute Module" to launch the curation interface

#### Curation Interface
The M3 window shows:
- **3D Mask Viewer**: Current mask displayed in 3D
- **DFF Trace Preview**: Calcium activity trace for current mask
- **Navigation**: Previous/Next buttons to browse masks
- **Actions**: Keep, Delete, Merge buttons

#### Navigation Controls
- **Next**: Move to next mask in sequence
- **Previous**: Move to previous mask
- **Jump to Mask**: Enter mask number directly (future feature)

#### Curation Actions

**Keep**: Mark mask as good quality
- Mask will be included in final analysis
- Trace shows clear calcium activity
- Structure looks like a dendrite

**Delete**: Remove mask from analysis
- Use for noise, artifacts, or poor segmentation
- Mask will not appear in downstream analysis
- Cannot be undone (within session)

**Merge**: Combine multiple masks
1. Select first mask and click "Merge"
2. Navigate to second mask
3. Click "Merge" again to combine
4. Merged mask gets new ID

#### DFF Trace Interpretation
The trace preview helps distinguish real activity from noise:

**Good Traces** (Keep):
- Clear peaks above baseline
- Consistent amplitude
- Temporal structure (not random)
- Low baseline noise

**Bad Traces** (Delete):
- Flat line (no activity)
- Pure noise (random fluctuations)
- Extreme outliers
- Negative-going only

#### Saving Progress
- Click "Save" to save curation decisions
- Click "Exit" to close without saving (confirmation required)
- Progress is saved to disk
- Can resume curation later from where you left off

#### Tips
- Review all masks before making final decisions
- Use trace preview as primary quality indicator
- When in doubt, keep the mask (can filter later)
- Merge masks that are clearly parts of the same dendrite
- Take breaks - curation can be time-consuming

### Output Verification

After each module completes, the UI verifies expected outputs:

#### Output Indicators
- **✓ Green**: File or directory exists
- **✗ Red**: File or directory missing
- **Path**: Shows relative path from data directory

#### Common Output Issues

**Missing Outputs After Success**
- Script may have failed silently
- Check console output for errors
- Verify input files exist
- Check disk space

**Partial Outputs**
- Some scripts create multiple files
- One file missing may indicate partial failure
- Review console for warnings

**Opening Output Directory**
- Click "Open Directory" button
- Opens data directory in file browser
- Manually inspect files
- Useful for debugging

### Progress Tracking

#### Per-Dataset Progress
- Progress is tracked separately for each dataset
- Switching datasets loads that dataset's progress
- Completed modules shown in green
- Progress persisted across application restarts

#### Progress State Location
- Stored in: `~/.mavca_ui/progress.json`
- Human-readable JSON format
- Can be manually edited if needed
- Backed up automatically

#### Resetting Progress
To reset progress for current dataset:
1. Close the application
2. Edit `~/.mavca_ui/progress.json`
3. Remove the dataset entry
4. Restart application

Or use command-line:
```bash
python main.py --reset-config
```

### Multi-Dataset Workflow

Working with multiple datasets:

#### Switching Datasets
1. Change DATE, MOUSE, or RUN parameters
2. Click "Apply"
3. Progress for new dataset loads automatically
4. Previous dataset progress is preserved

#### Dataset Organization
- Each dataset has independent progress
- Outputs stored in separate directories
- Can work on multiple datasets in parallel
- Switch freely between datasets

#### Tips
- Use consistent naming conventions
- Document dataset parameters
- Keep notes on curation decisions
- Back up progress.json periodically

### Viewing Analysis Results

#### Plot Viewer (M4, M5)
After M4 and M5 complete:
1. Toggle plot viewer: `Ctrl+Shift+P`
2. Select plot type from dropdown
3. Available plots:
   - DFF traces for individual masks
   - Depth analysis plots
   - Outside mask dynamics
   - Summary statistics

#### Plot Controls
- **Zoom**: Mouse wheel or zoom buttons
- **Pan**: Click and drag
- **Reset**: Reset to original view
- **Save**: Export plot to file (future feature)

### Troubleshooting

#### Application Won't Start

**Missing Dependencies**
```bash
pip install -r requirements_ui.txt
```

**Wrong Python Version**
```bash
python --version  # Should be 3.10 or higher
```

**Virtual Environment Not Activated**
```bash
source .venv311/bin/activate
```

#### Configuration Issues

**Invalid DATE Format**
- Must be YYYY-MM-DD
- Example: 2025-12-25, not 12/25/2025

**Invalid MOUSE/RUN**
- Only letters, numbers, underscores, hyphens
- No spaces or special characters

**Data Directory Not Found**
- Check base directory path
- Verify DATE/MOUSE/RUN values
- Ensure directory exists on disk

#### Execution Errors

**Script Not Found**
- Verify script paths in module definitions
- Check that code/ directory exists
- Ensure scripts haven't been moved

**Python Interpreter Not Found**
- Verify virtual environment exists
- Check .venv311 or .venv directory
- Reinstall virtual environment if needed

**Script Fails with Error**
- Check console output for details
- Verify input files exist
- Check script-specific parameters
- Review script documentation

**Missing Input Files**
- Run previous modules first
- Check expected outputs from previous modules
- Verify data directory structure

#### M3 Curation Issues

**Curation Window Won't Open**
- Ensure M2 is completed
- Check that labelmaps directory exists
- Verify Napari is installed (optional dependency)

**No Masks to Curate**
- M2 may have found no masks
- Check M2 output in console
- Review M2 parameters (thresholds)

**Trace Preview Not Showing**
- Check that preprocessed data exists
- Verify DFF computation completed
- Review console for errors

#### Performance Issues

**Slow Execution**
- Large datasets take time
- Check system resources (CPU, memory)
- Close other applications
- Consider running scripts directly for very large datasets

**UI Freezing**
- Script execution blocks UI (by design)
- Wait for completion
- Check console for progress
- Future versions will improve responsiveness

#### Data Issues

**Corrupted State Files**
- Delete `~/.mavca_ui/config.json`
- Delete `~/.mavca_ui/progress.json`
- Restart application

**Lost Progress**
- Check `~/.mavca_ui/progress.json`
- Restore from backup if available
- Re-run modules if necessary

### Getting Help

#### Log Files
- Location: `~/.mavca_ui/app.log`
- Contains detailed error messages
- Include in bug reports

#### Debug Mode
```bash
python main.py --debug
```
- Enables verbose logging
- Shows detailed execution information
- Useful for troubleshooting

#### Common Questions

**Q: Can I run modules out of order?**
A: You can navigate to any module, but execution should follow the sequence M1→M1.5→M2→M2.5→M3→M4→M5→M6 for correct results.

**Q: Can I edit script parameters?**
A: Currently, script-specific parameters must be edited in the Python files directly. The UI shows current values but doesn't allow editing yet.

**Q: Can I run multiple modules at once?**
A: No, only one module can execute at a time. This prevents conflicts and ensures proper sequencing.

**Q: What happens if I close the app during execution?**
A: The running script will be terminated. Progress is only saved after successful completion.

**Q: Can I use this on Windows or Linux?**
A: The UI is designed for macOS but should work on Linux and Windows with minor adjustments. Testing on these platforms is ongoing.

**Q: How do I back up my progress?**
A: Copy `~/.mavca_ui/progress.json` to a safe location. This file contains all progress data.

**Q: Can I share configurations between users?**
A: Yes, copy the config.json file. Progress is dataset-specific and can also be shared.

### Advanced Usage

#### Command-Line Options

**Reset Configuration**
```bash
python main.py --reset-config
```
Removes saved configuration, starts fresh.

**Debug Logging**
```bash
python main.py --debug
```
Enables detailed logging for troubleshooting.

#### Configuration Files

**Config Location**: `~/.mavca_ui/config.json`
```json
{
  "date": "2025-12-25",
  "mouse": "rAi162_phpeb",
  "run": "run1",
  "base_dir": "/path/to/data"
}
```

**Progress Location**: `~/.mavca_ui/progress.json`
```json
{
  "2025-12-25_rAi162_phpeb_run1": {
    "completed_modules": ["M1", "M2", "M3"],
    "m2_guided": true,
    "last_updated": "2025-01-15T14:30:00"
  }
}
```

#### Extending the UI

The UI is designed for extensibility:
- Add new modules in `ui/models/pipeline_modules.py`
- Create custom views in `ui/views/`
- Extend controller logic in `ui/controllers/`
- See developer documentation for details

### Best Practices

#### Workflow Organization
1. Use consistent naming conventions for datasets
2. Document parameter choices
3. Keep notes on curation decisions
4. Back up progress regularly
5. Verify outputs after each module

#### Data Management
1. Organize data by date/mouse/run
2. Keep raw data separate from processed
3. Document any manual interventions
4. Archive completed datasets

#### Quality Control
1. Review console output for warnings
2. Verify expected outputs exist
3. Spot-check intermediate results
4. Use M3 curation carefully
5. Document any issues or anomalies

### Future Features

Planned enhancements:
- **Script Parameter Editing**: Edit parameters directly in UI
- **Batch Processing**: Process multiple datasets automatically
- **Enhanced M6 Visualization**: Interactive 3D movie viewer
- **Behavioral Data Integration**: Pupil tracking, whisking, accelerometer
- **Custom Workflows**: Define custom module sequences
- **Remote Execution**: Run scripts on remote servers
- **Collaboration**: Share configurations and progress with team

## Project Structure

```
ui/
├── __init__.py              # Package initialization
├── models/                  # Data models and business logic
│   ├── __init__.py
│   ├── config.py           # PipelineConfig
│   ├── progress.py         # ProgressState
│   ├── module.py           # ModuleDefinition, ScriptInfo
│   ├── executor.py         # ScriptExecutor
│   └── state_manager.py    # StateManager
├── views/                   # UI components
│   ├── __init__.py
│   ├── main_window.py      # MainWindow
│   ├── parameter_panel.py  # ParameterPanel
│   ├── navigation_bar.py   # ModuleNavigationBar
│   ├── module_detail.py    # ModuleDetailView
│   ├── console_output.py   # ConsoleOutputWidget
│   ├── curation_window.py  # CurationWindow
│   └── plot_viewer.py      # PlotViewerWidget
├── controllers/             # Application logic
│   ├── __init__.py
│   ├── pipeline_controller.py  # PipelineController
│   └── script_runner.py        # ScriptRunner
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py         # Pytest configuration and fixtures
│   ├── test_models/        # Model layer tests
│   ├── test_views/         # View layer tests
│   ├── test_controllers/   # Controller layer tests
│   ├── test_properties/    # Property-based tests (Hypothesis)
│   └── test_integration/   # Integration tests
├── pytest.ini              # Pytest configuration
└── README.md               # This file
```

## Architecture

The application follows the Model-View-Controller (MVC) pattern:

- **Models**: Core data structures and business logic
  - Configuration management
  - Progress tracking
  - Module definitions
  - Script execution
  - State persistence

- **Views**: PyQt6-based UI components
  - Main application window
  - Parameter configuration panel
  - Module navigation and details
  - Console output display
  - M3 curation interface
  - Plot viewer

- **Controllers**: Application coordination
  - Pipeline execution orchestration
  - State management
  - Event handling

## Running the Application

### Basic Launch

From the project root directory:

```bash
# Activate virtual environment first
source .venv311/bin/activate  # or .venv/bin/activate

# Launch the application
python main.py
```

### Command-Line Options

```bash
# Launch with debug logging
python main.py --debug

# Reset configuration to defaults
python main.py --reset-config

# Show help
python main.py --help
```

### Alternative Launch Methods

```bash
# Using Python module syntax
python -m ui.main

# Make main.py executable (Unix/macOS)
chmod +x main.py
./main.py
```

## Running Tests

```bash
# Run all tests
pytest ui/tests/

# Run only unit tests
pytest ui/tests/ -m unit

# Run only property-based tests
pytest ui/tests/ -m property

# Run with verbose output
pytest ui/tests/ -v

# Run with coverage
pytest ui/tests/ --cov=ui --cov-report=html
```

## Development

### Adding New Features

1. Define models in `ui/models/`
2. Create views in `ui/views/`
3. Implement controllers in `ui/controllers/`
4. Write tests in `ui/tests/`

### Testing Strategy

The project uses a dual testing approach:

1. **Unit Tests**: Test specific examples and edge cases
   - Located in `test_models/`, `test_views/`, `test_controllers/`
   - Use pytest fixtures and mocks

2. **Property-Based Tests**: Test universal properties across all inputs
   - Located in `test_properties/`
   - Use Hypothesis for randomized testing
   - Minimum 100 iterations per property

3. **Integration Tests**: Test end-to-end workflows
   - Located in `test_integration/`
   - Test complete pipeline execution scenarios

## Requirements

See `requirements_ui.txt` for the complete list of dependencies:
- PyQt6 >= 6.6.0 (GUI framework)
- pytest >= 7.4.0 (testing)
- hypothesis >= 6.92.0 (property-based testing)
- matplotlib >= 3.8.0 (plotting)
- numpy >= 1.24.0 (data handling)
- napari >= 0.4.18 (3D visualization, optional)

## Platform Support

Currently supports:
- macOS (primary target)
- Linux (planned)
- Windows (planned)

## License

[Add license information]

## Contributing

[Add contribution guidelines]
