"""
Pytest configuration and shared fixtures for MAVCA Pipeline UI tests.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import sys

# Ensure PyQt5 is available for testing
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
except ImportError:
    pytest.skip("PyQt5 not available", allow_module_level=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "date": "2025-12-25",
        "mouse": "rAi162_phpeb",
        "run": "run1",
        "base_dir": "/Users/test/data"
    }


@pytest.fixture
def sample_progress_data():
    """Sample progress state data for testing."""
    return {
        "dataset_key": "2025-12-25_rAi162_phpeb_run1",
        "completed_modules": ["M1", "M1.5", "M2"],
        "m2_guided": False,
        "last_updated": "2025-01-15T14:30:00"
    }


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    # Don't quit the app as it may be used by other tests


@pytest.fixture
def qtbot(qapp):
    """Provide QtBot for testing Qt widgets."""
    try:
        from pytestqt.qtbot import QtBot
        bot = QtBot(qapp)
        yield bot
    except ImportError:
        # If pytest-qt is not installed, provide a minimal mock
        class MinimalQtBot:
            def wait(self, ms):
                qapp.processEvents()
                import time
                time.sleep(ms / 1000.0)
            
            def waitSignal(self, signal, timeout=1000):
                class SignalBlocker:
                    def __init__(self, signal):
                        self.signal = signal
                        self.args = []
                        self.signal.connect(self._slot)
                    
                    def _slot(self, *args):
                        self.args = list(args)
                    
                    def __enter__(self):
                        return self
                    
                    def __exit__(self, *args):
                        pass
                
                return SignalBlocker(signal)
        
        yield MinimalQtBot()


# Configure Hypothesis for property-based tests
from hypothesis import settings, Verbosity

# Register a profile for CI/CD with more iterations
settings.register_profile("ci", max_examples=100, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=20, verbosity=Verbosity.normal)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# Load the appropriate profile based on environment
import os
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
