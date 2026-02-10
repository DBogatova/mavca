# Implementation Plan: MAVCA Pipeline UI

## Overview

This implementation plan creates a PyQt6-based desktop application for the MAVCA calcium imaging analysis pipeline. The UI provides parameter configuration, script execution, progress tracking, and interactive mask curation. The implementation follows an MVC architecture with JSON-based state persistence and comprehensive testing using both unit tests and property-based tests with Hypothesis.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create `ui/` directory in project root
  - Create subdirectories: `ui/models/`, `ui/views/`, `ui/controllers/`, `ui/tests/`
  - Create `requirements_ui.txt` with PyQt6, Hypothesis, pytest dependencies
  - Create `ui/__init__.py` and package structure
  - _Requirements: 9.1, 9.2, 9.3_

- [x] 2. Implement core model classes
  - [x] 2.1 Implement PipelineConfig class
    - Write configuration management with DATE, MOUSE, RUN, base_dir
    - Implement validate_date() for YYYY-MM-DD format validation
    - Implement validate_identifier() for alphanumeric/underscore/hyphen validation
    - Implement get_data_path() for path construction
    - Implement serialization methods (to_dict, from_dict)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 11.1_
  
  - [x] 2.2 Write property test for date validation
    - **Property 5: Date Format Validation**
    - **Validates: Requirements 2.2**
  
  - [x] 2.3 Write property test for identifier validation
    - **Property 6: Identifier Validation**
    - **Validates: Requirements 2.3, 2.4**
  
  - [x] 2.4 Write property test for configuration persistence
    - **Property 7: Configuration Persistence Round-Trip**
    - **Validates: Requirements 2.5**
  
  - [x] 2.5 Write property test for path construction
    - **Property 27: Output Path Construction**
    - **Validates: Requirements 7.5, 11.1, 11.2**

- [x] 3. Implement progress tracking
  - [x] 3.1 Implement ProgressState class
    - Write progress state management with completed_modules set
    - Implement mark_complete(), is_complete(), reset() methods
    - Add m2_guided flag tracking
    - Implement serialization methods
    - _Requirements: 6.1, 6.2, 4.6_
  
  - [x] 3.2 Write property test for progress tracking
    - **Property 19: Progress State Tracking**
    - **Validates: Requirements 6.1**
  
  - [x] 3.3 Write property test for progress persistence
    - **Property 21: Progress Persistence Round-Trip**
    - **Validates: Requirements 6.4**
  
  - [x] 3.4 Write property test for progress reset
    - **Property 22: Progress Reset**
    - **Validates: Requirements 6.5**

- [x] 4. Implement state management
  - [x] 4.1 Implement StateManager class
    - Write state persistence to JSON files
    - Implement save_config() and load_config()
    - Implement save_progress() and load_progress()
    - Implement list_datasets() for dataset enumeration
    - Handle corrupted state files gracefully
    - _Requirements: 2.5, 6.4, 6.5, 6.6_
  
  - [x] 4.2 Write property test for dataset-specific progress loading
    - **Property 23: Dataset-Specific Progress Loading**
    - **Validates: Requirements 6.6**
  
  - [x] 4.3 Write unit tests for state manager error handling
    - Test corrupted JSON handling
    - Test missing file handling
    - Test permission errors

- [x] 5. Implement module definitions
  - [x] 5.1 Create ModuleDefinition and ScriptInfo dataclasses
    - Define module metadata structure
    - Define script information structure
    - Implement get_script_by_name() method
    - _Requirements: 1.1, 1.2, 1.3, 1.5_
  
  - [x] 5.2 Create MODULES configuration dictionary
    - Define all 8 modules (M1, M1.5, M2, M2.5, M3, M4, M5, M6)
    - Specify scripts for each module with paths and descriptions
    - Specify expected outputs for each module
    - Mark optional modules and scripts
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 7.1_
  
  - [x] 5.3 Write property test for module information completeness
    - **Property 1: Module Information Completeness**
    - **Validates: Requirements 1.2**
  
  - [x] 5.4 Write property test for script list completeness
    - **Property 4: Script List Completeness**
    - **Validates: Requirements 1.5**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement script execution
  - [x] 7.1 Implement ScriptExecutor class
    - Write subprocess-based script execution
    - Implement real-time output capture with callbacks
    - Implement terminate() for stopping scripts
    - Implement _inject_parameters() for parameter override
    - Find and use virtual environment Python interpreter
    - _Requirements: 3.1, 3.2, 3.6, 3.7, 9.2_
  
  - [x] 7.2 Write property test for parameter injection
    - **Property 13: Parameter Injection**
    - **Validates: Requirements 3.7**
  
  - [x] 7.3 Write unit tests for script execution
    - Test output capture with mock scripts
    - Test termination functionality
    - Test virtual environment detection

- [x] 8. Implement main controller
  - [x] 8.1 Implement PipelineController class
    - Write controller initialization and state loading
    - Implement update_config() for parameter changes
    - Implement execute_module() for script execution
    - Implement verify_outputs() for output checking
    - Implement launch_curation() for M3 window
    - Implement open_output_directory() for file browser
    - _Requirements: 2.6, 3.1, 3.4, 3.5, 5.1, 7.2, 7.4, 8.2_
  
  - [x] 8.2 Write property test for successful execution updates progress
    - **Property 11: Successful Execution Updates Progress**
    - **Validates: Requirements 3.4, 6.2**
  
  - [x] 8.3 Write property test for failed execution preserves progress
    - **Property 12: Failed Execution Preserves Progress**
    - **Validates: Requirements 3.5**
  
  - [x] 8.4 Write property test for execution exclusivity
    - **Property 10: Execution Exclusivity**
    - **Validates: Requirements 3.3**

- [x] 9. Implement parameter panel view
  - [x] 9.1 Create ParameterPanel widget
    - Create input fields for DATE, MOUSE, RUN
    - Add validation indicators for each field
    - Add base directory configuration button
    - Connect input changes to controller
    - Display validation errors inline
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 11.5_
  
  - [x] 9.2 Write unit tests for parameter panel
    - Test input field presence
    - Test validation feedback display
    - Test base directory configuration

- [x] 10. Implement module navigation bar
  - [x] 10.1 Create ModuleNavigationBar widget
    - Create tab/button for each module
    - Implement visual indication of current module
    - Implement visual distinction for completed modules
    - Implement visual indication for optional modules
    - Connect navigation to controller
    - _Requirements: 1.1, 1.3, 1.4, 6.3, 8.1, 8.3_
  
  - [x] 10.2 Write property test for current module indication
    - **Property 3: Current Module Visual Indication**
    - **Validates: Requirements 1.4**
  
  - [x] 10.3 Write property test for completion status distinction
    - **Property 20: Completion Status Visual Distinction**
    - **Validates: Requirements 6.3**
  
  - [x] 10.4 Write property test for unrestricted navigation
    - **Property 29: Unrestricted Module Navigation**
    - **Validates: Requirements 8.3**

- [x] 11. Implement module detail view
  - [x] 11.1 Create ModuleDetailView widget
    - Display module name, description, and purpose
    - Display list of scripts with descriptions
    - Display expected outputs with existence indicators
    - Add execution button for module
    - Add "Open Directory" button
    - Show script parameters with categorization
    - _Requirements: 1.2, 1.5, 7.1, 7.2, 7.3, 7.4, 12.1, 12.2, 12.3_
  
  - [x] 11.2 Write property test for expected outputs display
    - **Property 24: Expected Outputs Display**
    - **Validates: Requirements 7.1**
  
  - [x] 11.3 Write property test for output existence verification
    - **Property 25: Output Existence Verification**
    - **Validates: Requirements 7.2**
  
  - [x] 11.4 Write property test for missing output warning
    - **Property 26: Missing Output Warning**
    - **Validates: Requirements 7.3**
  
  - [x] 11.5 Write property test for script parameters display
    - **Property 36: Script Parameters Display**
    - **Validates: Requirements 12.1, 12.2**

- [x] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement console output widget
  - [x] 13.1 Create ConsoleOutputWidget
    - Create scrollable text area for output
    - Implement append_output() for real-time updates
    - Add auto-scroll to bottom functionality
    - Add clear button
    - Style with monospace font
    - _Requirements: 3.2_
  
  - [x] 13.2 Write unit tests for console output
    - Test output appending
    - Test auto-scroll behavior

- [x] 14. Implement main window
  - [x] 14.1 Create MainWindow class
    - Assemble all widgets into main layout
    - Implement menu bar with File, View, Help menus
    - Implement keyboard shortcuts
    - Implement window geometry persistence
    - Connect all signals to controller
    - Implement error and warning dialogs
    - _Requirements: 8.4, 10.6, 10.7_
  
  - [x] 14.2 Write property test for navigation preserves configuration
    - **Property 30: Navigation Preserves Configuration**
    - **Validates: Requirements 8.5**
  
  - [x] 14.3 Write property test for window geometry persistence
    - **Property 33: Window Geometry Persistence Round-Trip**
    - **Validates: Requirements 10.7**
  
  - [x] 14.4 Write unit tests for main window
    - Test all UI elements present
    - Test keyboard shortcuts
    - Test menu items

- [x] 15. Implement M2/M2.5 workflow logic
  - [x] 15.1 Add M2 completion dialog
    - Show options: "Proceed to M3" or "Refine with M2.5"
    - Implement guided labelmap tracking
    - Show comparison option if both guided and non-guided exist
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 15.2 Write property test for guided labelmap tracking
    - **Property 14: Guided Labelmap Tracking**
    - **Validates: Requirements 4.6, 6.7**
  
  - [x] 15.3 Write unit tests for M2/M2.5 workflow
    - Test M2 completion dialog
    - Test M2.5 guide drawing notice
    - Test M2 re-run with guides option

- [x] 16. Implement M3 curation window
  - [x] 16.1 Create CurationWindow class
    - Create window layout with mask viewer and trace plot areas
    - Implement mask loading from labelmaps directory
    - Implement navigation buttons (Next, Previous)
    - Implement action buttons (Keep, Delete, Merge)
    - Implement Save and Exit functionality
    - _Requirements: 5.1, 5.2, 5.7, 5.8_
  
  - [x] 16.2 Implement mask visualization
    - Integrate Napari viewer or create custom 3D viewer
    - Display current mask in 3D
    - Implement mask highlighting
    - _Requirements: 5.1_
  
  - [x] 16.3 Implement DFF trace preview
    - Compute DFF trace for selected mask
    - Display trace using matplotlib embedded widget
    - Add visual indicators for activity vs noise
    - Update trace when mask selection changes
    - _Requirements: 5.3_
  
  - [x] 16.4 Write property test for mask selection displays trace
    - **Property 15: Mask Selection Displays Trace**
    - **Validates: Requirements 5.3**
  
  - [x] 16.5 Write property test for mask navigation completeness
    - **Property 16: Mask Navigation Completeness**
    - **Validates: Requirements 5.5**
  
  - [x] 16.6 Write property test for curation operation updates state
    - **Property 17: Curation Operation Updates State**
    - **Validates: Requirements 5.6**
  
  - [x] 16.7 Write property test for curation state persistence
    - **Property 18: Curation State Persistence Round-Trip**
    - **Validates: Requirements 5.8**

- [x] 17. Implement plot viewer widget
  - [x] 17.1 Create PlotViewerWidget
    - Embed matplotlib figure widget
    - Implement plot loading from file paths
    - Implement zoom and pan controls
    - Add plot type selector (DFF traces, depth analysis, outside mask)
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  
  - [x] 17.2 Write unit tests for plot viewer
    - Test plot loading
    - Test zoom/pan functionality
    - Test plot type switching

- [x] 18. Implement error handling and user feedback
  - [x] 18.1 Add comprehensive error handling
    - Implement user-friendly error messages
    - Add data directory existence checking
    - Add missing input file warnings
    - Add missing output warnings
    - Implement visual feedback for all actions
    - _Requirements: 10.3, 11.3, 11.4_
  
  - [x] 18.2 Write property test for error message display
    - **Property 32: Error Message Display**
    - **Validates: Requirements 10.3**
  
  - [x] 18.3 Write property test for data directory check
    - **Property 34: Data Directory Existence Check**
    - **Validates: Requirements 11.3**
  
  - [x] 18.4 Write property test for missing directory warning
    - **Property 35: Missing Directory Warning**
    - **Validates: Requirements 11.4**

- [x] 19. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 20. Add macOS-specific features
  - [x] 20.1 Implement macOS integration
    - Set application icon and bundle identifier
    - Implement native file dialogs
    - Add macOS menu bar integration
    - Test with macOS-specific paths (spaces, Unicode)
    - _Requirements: 9.1, 9.4, 9.5_
  
  - [x] 20.2 Write property test for macOS path handling
    - **Property 31: macOS Path Handling**
    - **Validates: Requirements 9.5**

- [x] 21. Add future extensibility placeholders
  - [x] 21.1 Create placeholder for M6 visualization
    - Add basic M6 module support
    - Add placeholder for 3D interactive preview
    - Document extension points
    - _Requirements: 14.2, 14.5_
  
  - [x] 21.2 Create placeholder for behavioral analysis
    - Add placeholder for pupil tracking integration
    - Add placeholder for whisking data integration
    - Add placeholder for accelerometer data integration
    - Document extension points
    - _Requirements: 14.3, 14.5_
  
  - [x] 21.3 Write unit tests for extensibility
    - Test M6 placeholder exists
    - Test behavioral analysis placeholder exists

- [x] 22. Create application entry point
  - [x] 22.1 Create main.py launcher
    - Implement QApplication initialization
    - Implement controller and main window creation
    - Add command-line argument parsing (optional)
    - Add exception handling and logging
    - _Requirements: 9.1, 9.2_
  
  - [x] 22.2 Write integration tests
    - Test complete M1-M6 workflow
    - Test M2/M2.5 iteration workflow
    - Test multi-dataset switching
    - Test state persistence across restarts

- [x] 23. Create documentation
  - [x] 23.1 Write user documentation
    - Create README for UI usage
    - Document keyboard shortcuts
    - Document M3 curation workflow
    - Add troubleshooting guide
    - _Requirements: 10.1, 10.2_
  
  - [x] 23.2 Write developer documentation
    - Document architecture and design patterns
    - Document extension points for future features
    - Document testing strategy
    - Add code examples for common tasks
    - _Requirements: 14.5_

- [x] 24. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis
- Unit tests validate specific examples and edge cases
- The M3 curation interface is the most critical component for user workflow
- State persistence ensures users can resume work across sessions
- The design supports future extensions for M6 visualization and behavioral data
