#!/usr/bin/env python3
"""
MAVCA Pipeline UI - Main Entry Point

This script launches the MAVCA Pipeline UI application, providing a graphical
interface for the Mask-Assisted Volumetric Calcium Analysis pipeline.

Usage:
    python main.py [options]

Options:
    --debug         Enable debug logging
    --reset-config  Reset configuration to defaults
    --help          Show this help message

Requirements:
    - Python 3.10 or higher
    - PyQt6 and other dependencies (see requirements_ui.txt)
    - Virtual environment (.venv311 or .venv) in project root
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

from ui.controllers.pipeline_controller import PipelineController
from ui.views.main_window import MainWindow
from ui.utils.error_messages import ErrorMessages, WarningMessages
from ui.utils.macos_integration import setup_macos_integration
from ui.models.script_worker import ScriptWorker


def setup_logging(debug: bool = False) -> None:
    """Set up application logging.
    
    Args:
        debug: If True, set logging level to DEBUG, otherwise INFO
    """
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create log directory if it doesn't exist
    log_dir = Path.home() / ".mavca_ui"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log")
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("MAVCA Pipeline UI starting...")
    if debug:
        logger.debug("Debug logging enabled")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="MAVCA Pipeline UI - Graphical interface for calcium imaging analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                  # Launch with default settings
    python main.py --debug          # Launch with debug logging
    python main.py --reset-config   # Reset configuration and launch

For more information, see ui/README.md
        """
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--reset-config',
        action='store_true',
        help='Reset configuration to defaults before launching'
    )
    
    return parser.parse_args()


def reset_configuration() -> None:
    """Reset application configuration to defaults.
    
    Removes saved configuration and progress files, allowing the application
    to start fresh with default settings.
    """
    from ui.models.state_manager import StateManager
    
    state_dir = Path.home() / ".mavca_ui"
    state_manager = StateManager(state_dir)
    
    # Remove configuration file
    config_file = state_dir / "config.json"
    if config_file.exists():
        config_file.unlink()
        logging.info("Configuration file removed")
    
    # Note: We don't remove progress files as they contain valuable data
    # Users can manually delete progress.json if needed
    
    logging.info("Configuration reset complete")


def check_dependencies() -> bool:
    """Check if required dependencies are installed.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    missing_deps = []
    
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print("ERROR: Missing required dependencies:", file=sys.stderr)
        for dep in missing_deps:
            print(f"  - {dep}", file=sys.stderr)
        print("\nPlease install dependencies:", file=sys.stderr)
        print("  pip install -r requirements_ui.txt", file=sys.stderr)
        return False
    
    return True


def main() -> int:
    """Main application entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Reset configuration if requested
    if args.reset_config:
        logger.info("Resetting configuration...")
        try:
            reset_configuration()
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return 1
    
    # Create application
    try:
        # PyQt5 handles high DPI scaling automatically
        app = QApplication(sys.argv)
        app.setApplicationName("MAVCA Pipeline UI")
        app.setOrganizationName("MAVCA")
        app.setOrganizationDomain("mavca.org")
        
        logger.info("QApplication created")
        
        # Set up macOS integration
        setup_macos_integration(app)
        logger.info("macOS integration configured")
        
        # Create controller
        controller = PipelineController()
        logger.info("PipelineController created")
        
        # Initialize controller (loads saved state)
        try:
            controller.initialize()
            logger.info("Controller initialized")
        except Exception as e:
            logger.error(f"Failed to initialize controller: {e}")
            QMessageBox.critical(
                None,
                "Initialization Error",
                f"Failed to initialize application:\n{str(e)}\n\n"
                "The application will start with default settings."
            )
        
        # Create main window
        main_window = MainWindow()
        logger.info("MainWindow created")
        
        # Complete macOS integration with main window
        setup_macos_integration(app, main_window)
        
        # Connect controller to views
        _connect_controller_to_views(controller, main_window)
        logger.info("Controller connected to views")
        
        # Show main window
        main_window.show()
        logger.info("Main window shown")
        
        # Run application event loop
        exit_code = app.exec()
        logger.info(f"Application exiting with code {exit_code}")
        
        return exit_code
        
    except Exception as e:
        logger.exception("Fatal error during application startup")
        
        # Try to show error dialog if possible
        try:
            QMessageBox.critical(
                None,
                "Fatal Error",
                f"A fatal error occurred:\n{str(e)}\n\n"
                "Please check the log file for details."
            )
        except:
            pass
        
        return 1


def _connect_controller_to_views(controller: PipelineController, main_window: MainWindow) -> None:
    """Connect the controller to view components.
    
    Sets up signal/slot connections between the controller and UI widgets.
    
    Args:
        controller: Pipeline controller instance
        main_window: Main window instance
    """
    logger = logging.getLogger(__name__)
    
    # Get view components
    param_panel = main_window.get_parameter_panel()
    nav_bar = main_window.get_navigation_bar()
    detail_view = main_window.get_module_detail_view()
    
    # Initialize parameter panel with current config
    param_panel.set_date(controller.config.date)
    param_panel.set_mouse(controller.config.mouse)
    param_panel.set_run(controller.config.run)
    param_panel.set_base_dir(str(controller.config.base_dir))
    
    # Connect parameter changes
    def on_parameters_changed():
        date = param_panel.get_date()
        mouse = param_panel.get_mouse()
        run = param_panel.get_run()
        
        if controller.update_config(date, mouse, run):
            logger.info(f"Configuration updated: {date}/{mouse}/{run}")
            main_window.update_status(f"Dataset: {date}/{mouse}/{run}")
            
            # Update navigation bar to reflect new progress
            _update_navigation_progress(controller, nav_bar)
            
            # Check if data directory exists and warn if not
            data_path = controller.config.get_data_path()
            if not data_path.exists():
                main_window.show_warning(
                    "Data Directory Not Found",
                    WarningMessages.format_data_dir_not_exist(data_path)
                )
        else:
            logger.warning("Invalid configuration parameters")
            # Determine which parameter is invalid
            if not controller.config.validate_date(date):
                main_window.show_error(
                    "Invalid Date",
                    ErrorMessages.INVALID_DATE
                )
            elif not controller.config.validate_identifier(mouse):
                main_window.show_error(
                    "Invalid MOUSE",
                    ErrorMessages.INVALID_IDENTIFIER
                )
            elif not controller.config.validate_identifier(run):
                main_window.show_error(
                    "Invalid RUN",
                    ErrorMessages.INVALID_IDENTIFIER
                )
    
    param_panel.parameters_changed.connect(on_parameters_changed)
    
    # Connect module navigation
    def on_module_selected(module_id: str):
        logger.info(f"Module selected: {module_id}")
        module = controller.get_module(module_id)
        
        # Update detail view
        detail_view.set_module(module)
        # Note: set_config not implemented in ModuleDetailView
        
        # Check if module is complete
        is_complete = controller.is_module_complete(module_id)
        # Note: set_completion_status not implemented in ModuleDetailView
        
        # Verify outputs if complete
        if is_complete:
            output_status = controller.verify_outputs(module_id)
            detail_view.set_output_existence(output_status)
        
        main_window.update_status(f"Viewing: {module.name}")
    
    nav_bar.module_selected.connect(on_module_selected)
    
    # Connect module execution (threaded)
    # Keep a reference to the active worker so it isn't garbage-collected
    _active_worker = [None]  # mutable container so closures can write to it

    def on_execute_module(module_id: str):
        logger.info(f"Executing module: {module_id}")
        main_window.update_status(f"Executing {module_id}...")
        
        # Special handling for M3 - launch curation window
        if module_id == "M3":
            try:
                controller.launch_curation()
                main_window.update_status("M3 curation window opened")
                main_window.show_info(
                    "M3 Curation",
                    "The curation window has been opened.\n\n"
                    "Review each mask and use Keep/Delete/Merge buttons.\n"
                    "Save your progress before closing the window."
                )
            except FileNotFoundError as e:
                main_window.show_error(
                    "Cannot Launch Curation",
                    str(e)
                )
            except Exception as e:
                logger.error(f"Failed to launch curation: {e}")
                main_window.show_error(
                    "Curation Error",
                    f"Failed to launch curation window:\n{str(e)}"
                )
            return
        
        # Check if data directory exists
        data_path = controller.config.get_data_path()
        if not controller.config.base_dir.exists():
            main_window.show_error(
                "Base Directory Not Found",
                ErrorMessages.format_missing_base_dir(controller.config.base_dir)
            )
            return
        
        if not data_path.exists():
            proceed = main_window.ask_question(
                "Data Directory Not Found",
                WarningMessages.format_data_dir_not_exist(data_path) +
                "\n\nDo you want to proceed anyway?"
            )
            if not proceed:
                return
        
        # Clear console and start progress
        console = main_window.get_console_output()
        console.clear_output()
        
        module = controller.get_module(module_id)
        script_name = module.scripts[0].name if module.scripts else ""
        console.start_progress(script_name)
        
        # Disable execution during run
        detail_view.set_execution_enabled(False)
        
        # --- Threaded execution ---
        worker = ScriptWorker(controller, module_id)
        _active_worker[0] = worker

        # Each output line arrives on the main thread via signal
        worker.output_line.connect(main_window.append_console_output)

        def _on_success(completed_module_id: str):
            _active_worker[0] = None
            console.stop_progress(success=True)
            main_window.update_status(f"{completed_module_id} completed successfully")
            _update_navigation_progress(controller, nav_bar)
            on_module_selected(completed_module_id)
            _handle_module_completion(
                completed_module_id, controller, main_window, nav_bar, on_module_selected
            )
            detail_view.set_execution_enabled(True)

        def _on_error(failed_module_id: str, error_msg: str):
            _active_worker[0] = None
            logger.error(f"Module execution failed: {error_msg}")
            main_window.update_status(f"{failed_module_id} failed")

            # Stop progress bar
            if console.progress_bar.isVisible():
                console.stop_progress(success=False)

            # Show appropriate error dialog
            error_lower = error_msg.lower()
            if "virtual environment" in error_lower:
                main_window.show_error(
                    "Virtual Environment Not Found",
                    ErrorMessages.MISSING_VENV
                )
            elif "script not found" in error_lower:
                main_window.show_error(
                    "Script Not Found",
                    error_msg
                )
            elif "exit code" in error_lower:
                import re as _re
                match = _re.search(r'exit code (\d+)', error_msg)
                if match:
                    main_window.show_error(
                        "Script Execution Failed",
                        ErrorMessages.format_script_failed(int(match.group(1)))
                    )
                else:
                    main_window.show_error(
                        "Execution Failed",
                        f"Module {failed_module_id} failed:\n{error_msg}\n\n"
                        "Check the console output for details."
                    )
            else:
                main_window.show_error(
                    "Execution Failed",
                    f"Module {failed_module_id} failed:\n{error_msg}\n\n"
                    "Check the console output for details."
                )

            detail_view.set_execution_enabled(True)

        worker.finished_ok.connect(_on_success)
        worker.finished_err.connect(_on_error)
        worker.start()
    
    detail_view.execute_module.connect(on_execute_module)
    
    # Connect directory opening
    def on_open_directory(module_id: str):
        try:
            controller.open_output_directory(module_id)
            logger.info(f"Opened directory for {module_id}")
        except FileNotFoundError as e:
            main_window.show_warning(
                "Directory Not Found",
                ErrorMessages.format_missing_data_dir(controller.config.get_data_path())
            )
        except PermissionError as e:
            main_window.show_error(
                "Permission Denied",
                ErrorMessages.format_permission_error(controller.config.get_data_path())
            )
        except Exception as e:
            logger.error(f"Failed to open directory: {e}")
            main_window.show_error(
                "Error Opening Directory",
                ErrorMessages.format_unexpected_error(e)
            )
    
    detail_view.open_directory.connect(on_open_directory)
    
    # Initialize with first module
    nav_bar.set_current_module("M1")
    on_module_selected("M1")
    
    # Update navigation progress
    _update_navigation_progress(controller, nav_bar)
    
    logger.info("Controller-view connections established")


def _update_navigation_progress(controller: PipelineController, nav_bar) -> None:
    """Update navigation bar with current progress.
    
    Args:
        controller: Pipeline controller
        nav_bar: Navigation bar widget
    """
    completed_modules = set()
    for module_id in ["M1", "M1.5", "M2", "M2.5", "M3", "M4", "M5", "M6"]:
        if controller.is_module_complete(module_id):
            completed_modules.add(module_id)
    nav_bar.set_completed_modules(completed_modules)


def _handle_module_completion(
    module_id: str,
    controller: PipelineController,
    main_window: MainWindow,
    nav_bar,
    on_module_selected
) -> None:
    """Handle post-completion actions for specific modules.
    
    Args:
        module_id: The module that was completed
        controller: Pipeline controller
        main_window: Main window instance
        nav_bar: Navigation bar widget
        on_module_selected: Callback for module selection
    """
    from ui.views.m2_completion_dialog import M2CompletionDialog, M2_5CompletionDialog
    
    logger = logging.getLogger(__name__)
    
    if module_id == "M1":
        # Show results summary and activity timeline
        console = main_window.get_console_output()
        console.show_summary()
        
        # Load activity timeline in plot viewer
        data_path = controller.config.get_data_path()
        timeline_path = data_path / "preprocessed" / "activity_timeline.pdf"
        
        # Try PDF first, then SVG, then PNG
        if not timeline_path.exists():
            timeline_path = data_path / "preprocessed" / "activity_timeline.svg"
        if not timeline_path.exists():
            timeline_path = data_path / "preprocessed" / "activity_timeline.png"
        
        if timeline_path.exists():
            plot_viewer = main_window.get_plot_viewer()
            
            # Wire up the "View Plot" button
            console.view_plot_button.show()
            try:
                console.view_plot_button.clicked.disconnect()
            except TypeError:
                pass
            
            def show_timeline():
                plot_viewer.load_plot(timeline_path)
                main_window.show_plot_viewer()
            
            console.view_plot_button.clicked.connect(show_timeline)
            
            # Also auto-show it
            show_timeline()
            logger.info(f"Loaded activity timeline: {timeline_path}")
        
        main_window.show_info(
            "M1 Complete",
            f"Preprocessing and event detection finished.\n\n"
            f"The activity timeline is shown in the Plot Viewer below.\n"
            f"Check the Console for a results summary."
        )
    
    elif module_id == "M2":
        # Show M2 completion dialog
        labelmap_types = controller.check_labelmap_types()
        dialog = M2CompletionDialog(
            has_guided=labelmap_types["guided"],
            has_non_guided=labelmap_types["non_guided"],
            parent=main_window
        )
        
        if dialog.exec():
            action = dialog.get_selected_action()
            logger.info(f"M2 completion action: {action}")
            
            if action == M2CompletionDialog.ACTION_PROCEED_M3:
                # Navigate to M3
                nav_bar.set_current_module("M3")
                on_module_selected("M3")
                main_window.show_info(
                    "Ready for M3",
                    "Navigate to M3 to begin interactive mask curation."
                )
            elif action == M2CompletionDialog.ACTION_RUN_M2_5:
                # Navigate to M2.5
                nav_bar.set_current_module("M2.5")
                on_module_selected("M2.5")
                main_window.show_info(
                    "Ready for M2.5",
                    "Navigate to M2.5 to draw manual trunk guides.\n\n"
                    "After completing M2.5, return to M2 and run it again "
                    "to create guided labelmaps."
                )
            elif action == M2CompletionDialog.ACTION_COMPARE:
                # TODO: Implement comparison view
                main_window.show_info(
                    "Comparison View",
                    "Labelmap comparison feature coming soon.\n\n"
                    "For now, you can manually inspect the labelmaps in the output directory."
                )
    
    elif module_id == "M2.5":
        # Show M2.5 completion dialog
        dialog = M2_5CompletionDialog(parent=main_window)
        dialog.exec()
        
        # Navigate to M2
        nav_bar.set_current_module("M2")
        on_module_selected("M2")
    
    else:
        # Default completion message for other modules
        main_window.show_info(
            "Execution Complete",
            f"Module {module_id} completed successfully!"
        )


if __name__ == "__main__":
    sys.exit(main())
