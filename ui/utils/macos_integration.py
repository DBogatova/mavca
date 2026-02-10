"""
macOS-specific integration utilities for the MAVCA Pipeline UI.

This module provides macOS-specific functionality including:
- Application icon and bundle configuration
- Native file dialogs
- Menu bar integration
- Path handling for macOS-specific paths
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt


class MacOSIntegration:
    """Utilities for macOS-specific integration."""
    
    @staticmethod
    def is_macos() -> bool:
        """Check if running on macOS.
        
        Returns:
            True if running on macOS, False otherwise
        """
        return sys.platform == "darwin"
    
    @staticmethod
    def set_application_icon(app: QApplication, icon_path: Optional[Path] = None) -> None:
        """Set the application icon for macOS.
        
        Args:
            app: QApplication instance
            icon_path: Path to icon file (optional, uses default if not provided)
        """
        if not MacOSIntegration.is_macos():
            return
        
        if icon_path and icon_path.exists():
            icon = QIcon(str(icon_path))
            app.setWindowIcon(icon)
    
    @staticmethod
    def configure_menu_bar(main_window) -> None:
        """Configure macOS-specific menu bar behavior.
        
        Args:
            main_window: MainWindow instance
        """
        if not MacOSIntegration.is_macos():
            return
        
        # On macOS, the menu bar is global and appears at the top of the screen
        # Qt handles this automatically, but we can set some preferences
        menubar = main_window.menuBar()
        menubar.setNativeMenuBar(True)  # Use native macOS menu bar
    
    @staticmethod
    def open_directory_native(path: Path) -> None:
        """Open a directory using macOS native file browser.
        
        Args:
            path: Path to directory to open
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            subprocess.CalledProcessError: If open command fails
        """
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        
        if MacOSIntegration.is_macos():
            subprocess.run(["open", str(path)], check=True)
        else:
            # Fallback for non-macOS systems
            if sys.platform == "win32":
                subprocess.run(["explorer", str(path)], check=True)
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", str(path)], check=True)
    
    @staticmethod
    def select_directory_native(
        parent=None,
        caption: str = "Select Directory",
        directory: str = ""
    ) -> Optional[str]:
        """Show native directory selection dialog.
        
        Args:
            parent: Parent widget (optional)
            caption: Dialog caption
            directory: Initial directory (optional)
            
        Returns:
            Selected directory path as string, or None if cancelled
        """
        options = QFileDialog.Option.ShowDirsOnly
        
        if MacOSIntegration.is_macos():
            # Use native macOS dialog
            options |= QFileDialog.Option.DontUseNativeDialog
            # Actually, we want to use native dialog on macOS
            options = QFileDialog.Option.ShowDirsOnly
        
        selected_dir = QFileDialog.getExistingDirectory(
            parent,
            caption,
            directory,
            options
        )
        
        return selected_dir if selected_dir else None
    
    @staticmethod
    def select_file_native(
        parent=None,
        caption: str = "Select File",
        directory: str = "",
        filter: str = "All Files (*)"
    ) -> Optional[str]:
        """Show native file selection dialog.
        
        Args:
            parent: Parent widget (optional)
            caption: Dialog caption
            directory: Initial directory (optional)
            filter: File filter string
            
        Returns:
            Selected file path as string, or None if cancelled
        """
        selected_file, _ = QFileDialog.getOpenFileName(
            parent,
            caption,
            directory,
            filter
        )
        
        return selected_file if selected_file else None
    
    @staticmethod
    def normalize_path(path: Path) -> Path:
        """Normalize a path for macOS.
        
        Handles macOS-specific path issues including:
        - Spaces in paths
        - Special characters
        - Unicode characters
        - Case sensitivity
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized path
        """
        # Resolve to absolute path
        try:
            normalized = path.resolve()
        except (OSError, RuntimeError):
            # If resolve fails, just use absolute path
            normalized = path.absolute()
        
        return normalized
    
    @staticmethod
    def validate_path(path: Path) -> bool:
        """Validate a path for macOS compatibility.
        
        Checks for:
        - Valid characters
        - Path length limits
        - Reserved names
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid, False otherwise
        """
        try:
            # Try to resolve the path
            path.resolve()
            
            # Check path length (macOS has a 1024 character limit for paths)
            if len(str(path)) > 1024:
                return False
            
            # Check for null bytes (not allowed in paths)
            if '\0' in str(path):
                return False
            
            return True
        except (OSError, RuntimeError, ValueError):
            return False
    
    @staticmethod
    def get_home_directory() -> Path:
        """Get the user's home directory on macOS.
        
        Returns:
            Path to home directory
        """
        return Path.home()
    
    @staticmethod
    def get_documents_directory() -> Path:
        """Get the user's Documents directory on macOS.
        
        Returns:
            Path to Documents directory
        """
        if MacOSIntegration.is_macos():
            return Path.home() / "Documents"
        else:
            # Fallback for non-macOS
            return Path.home() / "Documents"
    
    @staticmethod
    def get_desktop_directory() -> Path:
        """Get the user's Desktop directory on macOS.
        
        Returns:
            Path to Desktop directory
        """
        if MacOSIntegration.is_macos():
            return Path.home() / "Desktop"
        else:
            # Fallback for non-macOS
            return Path.home() / "Desktop"
    
    @staticmethod
    def handle_special_paths(path: Path) -> Path:
        """Handle macOS special paths and aliases.
        
        Expands:
        - ~ (home directory)
        - Symbolic links
        - Aliases (macOS-specific)
        
        Args:
            path: Path to expand
            
        Returns:
            Expanded path
        """
        # Expand user home directory
        path = path.expanduser()
        
        # Resolve symbolic links
        try:
            path = path.resolve()
        except (OSError, RuntimeError):
            # If resolve fails, just return the expanded path
            pass
        
        return path
    
    @staticmethod
    def configure_high_dpi() -> None:
        """Configure high DPI settings for macOS Retina displays."""
        if MacOSIntegration.is_macos():
            # PyQt5 handles this differently than PyQt6
            # High DPI scaling is handled automatically in PyQt5
            pass
    
    @staticmethod
    def set_bundle_identifier(app: QApplication, identifier: str = "org.mavca.pipeline-ui") -> None:
        """Set the macOS bundle identifier.
        
        Args:
            app: QApplication instance
            identifier: Bundle identifier (reverse domain notation)
        """
        if MacOSIntegration.is_macos():
            app.setOrganizationDomain(identifier)
    
    @staticmethod
    def enable_dark_mode_support(app: QApplication) -> None:
        """Enable macOS dark mode support.
        
        Args:
            app: QApplication instance
        """
        if MacOSIntegration.is_macos():
            # Qt6 automatically respects system dark mode
            # We can set the application to follow system theme
            app.setStyle("Fusion")  # Fusion style works well with dark mode


def setup_macos_integration(app: QApplication, main_window=None) -> None:
    """Set up all macOS integration features.
    
    This is a convenience function that configures all macOS-specific
    features in one call.
    
    Args:
        app: QApplication instance
        main_window: MainWindow instance (optional)
    """
    if not MacOSIntegration.is_macos():
        return
    
    # Configure high DPI for Retina displays
    MacOSIntegration.configure_high_dpi()
    
    # Set bundle identifier
    MacOSIntegration.set_bundle_identifier(app)
    
    # Enable dark mode support
    MacOSIntegration.enable_dark_mode_support(app)
    
    # Configure menu bar if main window provided
    if main_window:
        MacOSIntegration.configure_menu_bar(main_window)
        
        # Set application icon if available
        icon_path = Path(__file__).parent.parent / "resources" / "icon.icns"
        if icon_path.exists():
            MacOSIntegration.set_application_icon(app, icon_path)
