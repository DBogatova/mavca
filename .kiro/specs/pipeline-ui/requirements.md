# Requirements Document

## Introduction

The MAVCA Pipeline UI is a graphical user interface for the Mask-Assisted Volumetric Calcium Analysis pipeline. The pipeline consists of sequential modules (M1 through M6) that process 4D calcium imaging data to extract and analyze dendritic calcium activity. The workflow includes an iterative refinement loop (M2-M2.5-M2) and a critical interactive curation step (M3) where researchers manually review and edit 3D masks. The UI will provide researchers with an intuitive way to configure parameters, execute scripts, track progress, and perform interactive curation without requiring deep technical knowledge of command-line operations.

## Glossary

- **Pipeline**: The complete MAVCA analysis workflow consisting of sequential modules with an iterative refinement loop
- **Module**: A logical grouping of one or more Python scripts that perform a specific analysis step (M1, M1.5, M2, M2.5, M3, M4, M5, M6)
- **Script**: A Python file that performs a specific computational task within a module
- **Common_Parameters**: The DATE, MOUSE, and RUN configuration values used across all pipeline scripts
- **UI**: The graphical user interface application
- **Progress_State**: The record of which modules have been completed for a given dataset
- **Expected_Output**: The files and directories that should be created after successfully running a module
- **Sequential_Execution**: The requirement that modules must be run in order (M1 before M2, etc.)
- **Curation_Interface**: The interactive M3 interface for reviewing, editing, and selecting 3D masks
- **Mask**: A 3D segmented region representing a dendrite structure
- **DFF_Trace**: Delta F over F calcium activity trace showing fluorescence changes over time
- **Labelmap**: A 3D volume where each voxel is labeled with a mask ID (0 for background, 1+ for masks)
- **Guided_Labelmap**: A labelmap created using manual trunk guides from M2.5
- **Non-Guided_Labelmap**: A labelmap created by automatic segmentation without manual guides

## Requirements

### Requirement 1: Module Display

**User Story:** As a researcher, I want to see all pipeline modules in a clear sequential interface, so that I understand the complete workflow and my current position in the pipeline.

#### Acceptance Criteria

1. THE UI SHALL display all 6 modules (M1, M1.5, M2, M2.5, M3, M4, M5, M6) in sequential order
2. WHEN displaying modules, THE UI SHALL show the module number, name, and purpose for each module
3. WHEN displaying modules, THE UI SHALL indicate which modules are optional (M1.5, M2.5, and optional scripts within M5)
4. THE UI SHALL provide a visual indication of the current module being viewed or executed
5. THE UI SHALL display the list of scripts belonging to each module

### Requirement 2: Common Parameter Configuration

**User Story:** As a researcher, I want to configure the DATE, MOUSE, and RUN parameters in one place, so that I don't have to edit multiple Python files manually.

#### Acceptance Criteria

1. THE UI SHALL provide input fields for DATE, MOUSE, and RUN parameters
2. WHEN a user enters DATE, THE UI SHALL validate it follows the YYYY-MM-DD format
3. WHEN a user enters MOUSE, THE UI SHALL accept alphanumeric strings with underscores and hyphens
4. WHEN a user enters RUN, THE UI SHALL accept alphanumeric strings with underscores and hyphens
5. THE UI SHALL persist the configured parameters across UI sessions
6. WHEN parameters are changed, THE UI SHALL indicate that scripts will use the new values on next execution

### Requirement 3: Script Execution

**User Story:** As a researcher, I want to run scripts for each module from the UI, so that I can execute the pipeline without using the command line.

#### Acceptance Criteria

1. WHEN a user selects a module, THE UI SHALL provide a way to execute the scripts for that module
2. WHEN a script is executing, THE UI SHALL display real-time output from the Python script
3. WHEN a script is executing, THE UI SHALL prevent the user from starting another script execution
4. WHEN a script completes successfully, THE UI SHALL indicate success and update the progress state
5. IF a script execution fails, THEN THE UI SHALL display the error message and maintain the current progress state
6. THE UI SHALL provide a way to stop a running script execution
7. WHEN executing scripts, THE UI SHALL use the configured DATE, MOUSE, and RUN parameters

### Requirement 4: M2/M2.5 Iteration Workflow

**User Story:** As a researcher, I want to iteratively refine mask segmentation by running M2, optionally running M2.5 to add guides, and re-running M2, so that I can improve segmentation quality.

#### Acceptance Criteria

1. WHEN M2 completes, THE UI SHALL provide options to proceed to M3 or run M2.5 for refinement
2. WHEN a user chooses to run M2.5, THE UI SHALL indicate that manual guide drawing is required
3. WHEN M2.5 completes, THE UI SHALL provide an option to re-run M2 with guides enabled
4. WHEN re-running M2 after M2.5, THE UI SHALL indicate that guided labelmaps will be created
5. THE UI SHALL allow comparing guided and non-guided labelmaps before proceeding to M3
6. THE UI SHALL track whether the current labelmaps are guided or non-guided

### Requirement 5: Interactive Mask Curation (M3)

**User Story:** As a researcher, I want to interactively review and edit 3D masks with navigation controls and real-time DFF trace previews, so that I can curate high-quality masks for analysis.

#### Acceptance Criteria

1. WHEN M3 is launched, THE UI SHALL display the current 3D mask with navigation controls
2. THE UI SHALL provide buttons for: Next, Previous, Keep, Delete, and Merge operations
3. WHEN a mask is selected, THE UI SHALL display a preview of the DFF trace for that mask
4. WHEN viewing a DFF trace preview, THE UI SHALL help the user distinguish real activity from noise
5. THE UI SHALL allow the user to navigate through all masks in the dataset
6. WHEN a user performs Keep, Delete, or Merge operations, THE UI SHALL update the mask state immediately
7. WHEN curation is complete, THE UI SHALL save the curated masks and update progress state
8. THE UI SHALL provide a way to exit curation and resume later without losing progress

### Requirement 6: Progress Tracking

**User Story:** As a researcher, I want to track which modules I have completed, so that I know where I am in the pipeline and what steps remain.

#### Acceptance Criteria

1. THE UI SHALL maintain a progress state indicating which modules have been completed
2. WHEN a module's scripts complete successfully, THE UI SHALL mark that module as completed
3. WHEN displaying modules, THE UI SHALL visually distinguish completed modules from incomplete modules
4. THE UI SHALL persist progress state across UI sessions for each unique combination of DATE, MOUSE, and RUN
5. THE UI SHALL provide a way to reset progress state for a dataset
6. WHEN a user changes DATE, MOUSE, or RUN parameters, THE UI SHALL load the progress state for that specific dataset
7. THE UI SHALL track whether M2 was run with or without guides

### Requirement 7: Expected Output Display

**User Story:** As a researcher, I want to see what outputs each module should produce, so that I can verify the module executed correctly and understand what data is available.

#### Acceptance Criteria

1. WHEN displaying a module, THE UI SHALL show the expected output files and directories for that module
2. WHEN a module is completed, THE UI SHALL indicate which expected outputs exist on the filesystem
3. IF expected outputs are missing after module completion, THEN THE UI SHALL provide a visual warning
4. THE UI SHALL provide a way to open the output directory for a module in the system file browser
5. THE UI SHALL display output paths using the configured DATE, MOUSE, and RUN parameters

### Requirement 8: Module Navigation

**User Story:** As a researcher, I want to easily navigate between modules, so that I can review previous steps or jump to specific modules.

#### Acceptance Criteria

1. THE UI SHALL provide navigation controls to move between modules
2. WHEN a user navigates to a module, THE UI SHALL display that module's details, scripts, and expected outputs
3. THE UI SHALL allow navigation to any module regardless of completion status
4. THE UI SHALL provide a way to return to an overview of all modules
5. WHEN navigating between modules, THE UI SHALL preserve the current parameter configuration

### Requirement 9: Platform Compatibility

**User Story:** As a researcher using macOS, I want the UI to work seamlessly on my system, so that I can use it without compatibility issues.

#### Acceptance Criteria

1. THE UI SHALL run on macOS systems
2. WHEN executing Python scripts, THE UI SHALL use the Python interpreter from the project's virtual environment
3. THE UI SHALL integrate with the existing Python codebase without requiring modifications to existing scripts
4. THE UI SHALL follow macOS user interface conventions and design patterns
5. THE UI SHALL handle macOS-specific file paths correctly

### Requirement 10: User Experience

**User Story:** As a researcher who may not be deeply technical, I want the UI to be simple and intuitive, so that I can focus on my research rather than learning complex tools.

#### Acceptance Criteria

1. THE UI SHALL use clear, non-technical language for labels and instructions
2. THE UI SHALL provide helpful tooltips or descriptions for configuration parameters
3. WHEN errors occur, THE UI SHALL display user-friendly error messages with suggested actions
4. THE UI SHALL provide visual feedback for all user actions within 100 milliseconds
5. THE UI SHALL use a clean, uncluttered layout that prioritizes essential information
6. THE UI SHALL provide keyboard shortcuts for common actions
7. THE UI SHALL remember window size and position across sessions

### Requirement 11: Data Path Management

**User Story:** As a researcher, I want the UI to automatically construct correct file paths based on my parameters, so that I don't have to worry about directory structure.

#### Acceptance Criteria

1. THE UI SHALL construct data paths following the pattern: data/DATE/MOUSE/RUN/
2. WHEN displaying expected outputs, THE UI SHALL use the constructed data path
3. WHEN executing scripts, THE UI SHALL verify the base data directory exists
4. IF the data directory for the configured parameters does not exist, THEN THE UI SHALL display a warning before script execution
5. THE UI SHALL provide a way to configure the base data directory location

### Requirement 12: Script Parameter Visibility

**User Story:** As a researcher, I want to see what other parameters each script uses, so that I understand what configuration options are available beyond DATE, MOUSE, and RUN.

#### Acceptance Criteria

1. WHEN displaying a script, THE UI SHALL show the key configuration parameters defined in that script
2. THE UI SHALL display parameter names, current values, and brief descriptions
3. THE UI SHALL indicate which parameters are common (DATE, MOUSE, RUN) and which are script-specific
4. THE UI SHALL provide a way to view the full script source code
5. THE UI SHALL indicate that script-specific parameters must be edited in the Python files directly

### Requirement 13: Analysis Visualization

**User Story:** As a researcher, I want to view plots and analysis results from M4 and M5 within the UI, so that I can quickly assess data quality without opening external files.

#### Acceptance Criteria

1. WHEN M4 or M5 completes, THE UI SHALL display generated plots and analysis results
2. THE UI SHALL provide a way to view DFF traces for individual masks
3. THE UI SHALL provide a way to view depth analysis plots
4. THE UI SHALL provide a way to view outside mask dynamics plots
5. THE UI SHALL allow zooming and panning of displayed plots

### Requirement 14: Future Extensibility

**User Story:** As a researcher, I want the UI to support future additions like 3D interactive previews and behavioral data integration, so that the tool can grow with the project needs.

#### Acceptance Criteria

1. THE UI SHALL be designed with a modular architecture that allows adding new modules
2. THE UI SHALL provide placeholder or basic support for M6 3D visualization
3. THE UI SHALL provide placeholder or basic support for behavioral analysis integration (pupil, whisking, accelerometer)
4. THE UI SHALL allow adding new script types without requiring major refactoring
5. THE UI SHALL document extension points for future developers

