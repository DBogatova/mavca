# MAVCA Pipeline UI - Developer Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Patterns](#design-patterns)
3. [Extension Points](#extension-points)
4. [Testing Strategy](#testing-strategy)
5. [Code Examples](#code-examples)
6. [Contributing Guidelines](#contributing-guidelines)

## Architecture Overview

The MAVCA Pipeline UI follows a Model-View-Controller (MVC) architecture pattern, providing clear separation of concerns and maintainability.

### Directory Structure

```
ui/
├── models/              # Data models and business logic
│   ├── pipeline_config.py
│   ├── progress_state.py
│   ├── state_manager.py
│   ├── script_executor.py
│   ├── module_definition.py
│   └── pipeline_modules.py
├── views/               # UI components
│   ├── main_window.py
│   ├── parameter_panel.py
│   ├── module_navigation_bar.py
│   ├── module_detail_view.py
│   ├── console_output_widget.py
│   ├── plot_viewer_widget.py
│   └── m2_completion_dialog.py
├── controllers/         # Application logic coordination
│   └── pipeline_controller.py
├── extensions/          # Extension points for future features
│   ├── m6_visualization.py
│   └── behavioral_analysis.py
├── utils/               # Utility modules
│   ├── error_messages.py
│   └── macos_integration.py
└── tests/               # Test suite
    ├── test_models/
    ├── test_views/
    ├── test_controllers/
    ├── test_properties/
    ├── test_integration/
    └── test_extensions/
```

### Component Responsibilities

#### Models
- **PipelineConfig**: Manages DATE, MOUSE, RUN, and base directory configuration
- **ProgressState**: Tracks module completion status per dataset
- **StateManager**: Handles JSON-based persistence of configuration and progress
- **ScriptExecutor**: Executes Python scripts with real-time output capture
- **ModuleDefinition**: Defines module metadata, scripts, and expected outputs

#### Views
- **MainWindow**: Primary application window with menu bar and layout
- **ParameterPanel**: Configuration input fields with validation
- **ModuleNavigationBar**: Module selection tabs with progress indicators
- **ModuleDetailView**: Displays module information and execution controls
- **ConsoleOutputWidget**: Real-time script output display
- **PlotViewerWidget**: Matplotlib plot embedding and interaction

#### Controllers
- **PipelineController**: Coordinates between models and views, manages application state

## Design Patterns

### 1. Model-View-Controller (MVC)

The application strictly separates data (models), presentation (views), and logic (controllers).

**Benefits:**
- Clear separation of concerns
- Easier testing (models can be tested independently)
- Flexible UI changes without affecting business logic

**Example:**
```python
# Model
config = PipelineConfig()
config.date = "2025-01-01"

# Controller
controller = PipelineController()
controller.update_config(date, mouse, run)

# View
param_panel.set_date(config.date)
```

### 2. Observer Pattern (Signals/Slots)

PyQt6's signal/slot mechanism is used for loose coupling between components.

**Example:**
```python
# View emits signal
class ParameterPanel(QWidget):
    parameters_changed = pyqtSignal()
    
    def on_date_changed(self):
        self.parameters_changed.emit()

# Controller connects to signal
param_panel.parameters_changed.connect(on_parameters_changed)
```

### 3. Strategy Pattern (Extension Points)

Extension points allow pluggable backends for visualization and analysis.

**Example:**
```python
# Register custom backend
extension = get_m6_extension()
extension.register_backend("custom", CustomVisualizationBackend())
extension.set_active_backend("custom")
```

### 4. Template Method Pattern (State Persistence)

State serialization follows a consistent pattern across all stateful classes.

**Example:**
```python
class ProgressState:
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {...}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressState':
        """Deserialize from dictionary"""
        return cls(...)
```

## Extension Points

### 1. M6 3D Visualization

The M6 visualization extension provides a framework for adding custom 3D visualization backends.

#### Creating a Custom Backend

```python
from ui.extensions import get_m6_extension

class CustomVisualizationBackend:
    """Custom 3D visualization backend."""
    
    def load_data(self, data_path: Path) -> bool:
        """Load 3D volume data."""
        # Implement data loading
        return True
    
    def render(self, widget: QWidget) -> None:
        """Render 3D visualization."""
        # Implement rendering
        pass

# Register the backend
extension = get_m6_extension()
extension.register_backend("custom", CustomVisualizationBackend())
extension.set_active_backend("custom")
```

#### Available Extension Points

- **Backend Registration**: Add custom visualization libraries (VTK, Napari, etc.)
- **Widget Customization**: Extend `M6VisualizationWidget` for custom UI
- **Data Loading**: Implement custom data loaders for different formats
- **Rendering**: Implement custom rendering pipelines

### 2. Behavioral Data Integration

The behavioral analysis extension provides a framework for integrating behavioral data sources.

#### Creating a Custom Processor

```python
from ui.extensions import get_behavioral_extension, BehavioralDataType

class PupilTrackingProcessor:
    """Process pupil tracking data."""
    
    def load_data(self, data_path: Path) -> bool:
        """Load pupil tracking data."""
        # Implement data loading
        return True
    
    def synchronize(self, imaging_timestamps: np.ndarray) -> np.ndarray:
        """Synchronize with imaging data."""
        # Implement synchronization
        return synchronized_data
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze pupil data."""
        # Implement analysis
        return results

# Register the processor
extension = get_behavioral_extension()
extension.register_processor("pupil", PupilTrackingProcessor())
extension.activate_processor("pupil")
```

#### Available Extension Points

- **Processor Registration**: Add custom behavioral data processors
- **Data Type Support**: Extend `BehavioralDataType` enum for new data types
- **Synchronization**: Implement custom synchronization algorithms
- **Analysis**: Add custom correlation and analysis methods

### 3. Custom Module Types

Add new module types to the pipeline by extending the module definitions.

#### Adding a New Module

```python
from ui.models.module_definition import ModuleDefinition, ScriptInfo
from ui.models.pipeline_modules import MODULES

# Define new module
new_module = ModuleDefinition(
    id="M7",
    name="Custom Analysis",
    description="Custom analysis module",
    scripts=[
        ScriptInfo(
            name="custom_analysis.py",
            path=Path("code/Custom/custom_analysis.py"),
            description="Run custom analysis",
            parameters={"PARAM1": 10, "PARAM2": "value"}
        )
    ],
    expected_outputs=[
        "custom_results/",
        "custom_plots/"
    ]
)

# Add to modules dictionary
MODULES["M7"] = new_module
```

### 4. Custom Error Messages

Extend error messages for custom scenarios.

```python
from ui.utils.error_messages import ErrorMessages

# Add custom error message
ErrorMessages.CUSTOM_ERROR = (
    "Custom error occurred:\n{details}\n\n"
    "Suggestions:\n"
    "• Check custom configuration\n"
    "• Verify custom inputs"
)

# Use custom error
message = ErrorMessages.CUSTOM_ERROR.format(details="Error details")
```

## Testing Strategy

### Test Organization

The test suite is organized into several categories:

1. **Unit Tests** (`test_models/`, `test_views/`, `test_controllers/`)
   - Test individual components in isolation
   - Mock dependencies
   - Fast execution

2. **Property-Based Tests** (`test_properties/`)
   - Test universal properties using Hypothesis
   - Generate random inputs
   - Verify invariants hold across all inputs

3. **Integration Tests** (`test_integration/`)
   - Test component interactions
   - Test complete workflows
   - Use temporary directories for state

4. **Extension Tests** (`test_extensions/`)
   - Test extension points
   - Verify extensibility mechanisms

### Running Tests

```bash
# Run all tests
cd ui
pytest

# Run specific test category
pytest tests/test_models/
pytest tests/test_properties/

# Run with coverage
pytest --cov=ui --cov-report=html

# Run property tests with more examples
pytest tests/test_properties/ --hypothesis-show-statistics
```

### Writing Tests

#### Unit Test Example

```python
def test_pipeline_config_validation():
    """Test configuration validation."""
    config = PipelineConfig()
    
    # Valid date
    assert config.validate_date("2025-01-01")
    
    # Invalid date
    assert not config.validate_date("2025-13-01")
```

#### Property Test Example

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=10, max_size=10))
def test_date_validation_property(date_string: str):
    """Test date validation for any string."""
    config = PipelineConfig()
    is_valid = config.validate_date(date_string)
    
    # Property: validation should match expected format
    expected_valid = _is_valid_date_format(date_string)
    assert is_valid == expected_valid
```

#### Integration Test Example

```python
def test_complete_workflow(temp_state_dir, temp_data_dir):
    """Test complete pipeline workflow."""
    state_manager = StateManager(temp_state_dir)
    
    # Configure
    config = PipelineConfig()
    config.date = "2025-01-01"
    state_manager.save_config(config)
    
    # Execute modules
    progress = ProgressState("2025-01-01_mouse_run")
    progress.mark_complete("M1")
    state_manager.save_progress(progress)
    
    # Verify persistence
    loaded = state_manager.load_progress("2025-01-01_mouse_run")
    assert loaded.is_complete("M1")
```

### Test Coverage Goals

- **Models**: 90%+ coverage
- **Controllers**: 85%+ coverage
- **Views**: 70%+ coverage (UI testing is more complex)
- **Overall**: 80%+ coverage

## Code Examples

### Example 1: Adding a Custom View Widget

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSignal

class CustomAnalysisWidget(QWidget):
    """Custom analysis widget."""
    
    analysis_requested = pyqtSignal(str)  # Signal for analysis request
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("<h2>Custom Analysis</h2>")
        layout.addWidget(title)
        
        # Add more widgets...
    
    def run_analysis(self, analysis_type: str):
        """Run custom analysis."""
        self.analysis_requested.emit(analysis_type)
```

### Example 2: Extending the Controller

```python
from ui.controllers.pipeline_controller import PipelineController

class ExtendedPipelineController(PipelineController):
    """Extended controller with custom functionality."""
    
    def __init__(self):
        super().__init__()
        self.custom_data = {}
    
    def run_custom_analysis(self, analysis_type: str):
        """Run custom analysis."""
        # Implement custom logic
        results = self._perform_analysis(analysis_type)
        self.custom_data[analysis_type] = results
        return results
    
    def _perform_analysis(self, analysis_type: str):
        """Perform the actual analysis."""
        # Implementation
        return {}
```

### Example 3: Custom State Persistence

```python
from ui.models.state_manager import StateManager
from pathlib import Path
import json

class CustomStateManager(StateManager):
    """Extended state manager with custom persistence."""
    
    def save_custom_data(self, data: dict) -> None:
        """Save custom data."""
        custom_file = self.state_dir / "custom_data.json"
        with open(custom_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_custom_data(self) -> dict:
        """Load custom data."""
        custom_file = self.state_dir / "custom_data.json"
        if custom_file.exists():
            with open(custom_file, 'r') as f:
                return json.load(f)
        return {}
```

### Example 4: Custom Script Executor

```python
from ui.models.script_executor import ScriptExecutor
from pathlib import Path

class CustomScriptExecutor(ScriptExecutor):
    """Extended script executor with custom features."""
    
    def execute_with_timeout(
        self,
        script_path: Path,
        config,
        output_callback,
        timeout: int = 300
    ) -> int:
        """Execute script with timeout."""
        # Implement timeout logic
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Script execution timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            return self.execute_script(script_path, config, output_callback)
        finally:
            signal.alarm(0)  # Cancel alarm
```

## Contributing Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public methods
- Keep functions focused and small (< 50 lines)

### Documentation

- Update docstrings when changing function signatures
- Add comments for complex logic
- Update this developer documentation for architectural changes
- Include examples in docstrings

### Testing

- Write tests for all new features
- Maintain or improve test coverage
- Run full test suite before committing
- Add property tests for validation logic

### Git Workflow

1. Create a feature branch from `main`
2. Make changes with clear commit messages
3. Run tests and ensure they pass
4. Update documentation
5. Submit pull request with description

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `style`: Code style changes

Example:
```
feat: Add M6 3D visualization extension point

- Created M6VisualizationExtension class
- Added backend registration system
- Implemented placeholder widget
- Added unit tests

Closes #123
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or documented)
- [ ] Error handling is appropriate
- [ ] Performance is acceptable

## Additional Resources

### PyQt6 Documentation
- [PyQt6 Reference](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [Qt6 Documentation](https://doc.qt.io/qt-6/)

### Testing Resources
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)

### Python Best Practices
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

## Support

For questions or issues:
1. Check this documentation
2. Review existing code examples
3. Check the test suite for usage examples
4. Open an issue on the project repository

## Version History

- **v1.0.0** (2025-01): Initial release with core functionality
  - M1-M6 module support
  - M2/M2.5 iteration workflow
  - State persistence
  - Extension points for M6 and behavioral analysis
