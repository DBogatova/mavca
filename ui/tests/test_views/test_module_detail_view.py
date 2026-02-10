"""
Property tests and unit tests for ModuleDetailView widget.

Tests expected outputs display, output existence verification,
missing output warning, and script parameters display.
"""

import pytest
from hypothesis import given, strategies as st
from pathlib import Path
from PyQt5.QtWidgets import QApplication

from ui.views.module_detail_view import ModuleDetailView
from ui.models.module_definition import ModuleDefinition, ScriptInfo


@pytest.fixture
def app(qapp):
    """Provide QApplication instance."""
    return qapp


@pytest.fixture
def module_detail_view(app):
    """Create a ModuleDetailView instance for testing."""
    return ModuleDetailView()


@pytest.fixture
def sample_module():
    """Create a sample module definition for testing."""
    return ModuleDefinition(
        id="M1",
        name="Test Module",
        description="A test module for unit testing",
        scripts=[
            ScriptInfo(
                name="test_script.py",
                path=Path("code/test_script.py"),
                description="Test script",
                parameters={"PARAM1": 10, "PARAM2": "value"}
            )
        ],
        expected_outputs=[
            "output1.txt",
            "output2.csv",
            "output_dir/"
        ]
    )


# Feature: pipeline-ui, Property 24: Expected Outputs Display
@given(
    st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10, unique=True)
)
def test_expected_outputs_display_property(output_paths):
    """
    For any module, the UI should display all expected output paths
    for that module.
    
    Validates: Requirements 7.1
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    detail_view = ModuleDetailView()
    
    # Create module with expected outputs
    module = ModuleDefinition(
        id="M_TEST",
        name="Test Module",
        description="Test",
        scripts=[],
        expected_outputs=output_paths
    )
    
    # Set module
    detail_view.set_module(module)
    
    # Verify all expected outputs are displayed
    # Check the outputs layout for labels containing each output path
    outputs_layout = detail_view.outputs_layout
    
    # Get all text from output labels
    displayed_outputs = []
    for i in range(outputs_layout.count()):
        item = outputs_layout.itemAt(i)
        if item.layout():
            # It's a nested layout (HBoxLayout with indicator and path)
            for j in range(item.layout().count()):
                widget = item.layout().itemAt(j).widget()
                if widget and hasattr(widget, 'text'):
                    text = widget.text()
                    if text and text not in ["✓", "✗"]:  # Skip indicators
                        displayed_outputs.append(text)
    
    # Verify all expected outputs are displayed
    for output_path in output_paths:
        assert any(output_path in displayed for displayed in displayed_outputs), (
            f"Expected output '{output_path}' should be displayed in the UI"
        )


# Feature: pipeline-ui, Property 25: Output Existence Verification
@given(
    st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10, unique=True),
    st.data()
)
def test_output_existence_verification_property(output_paths, data):
    """
    For any completed module, the UI should check and indicate which
    expected outputs exist on the filesystem.
    
    Validates: Requirements 7.2
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    detail_view = ModuleDetailView()
    
    # Create module with expected outputs
    module = ModuleDefinition(
        id="M_TEST",
        name="Test Module",
        description="Test",
        scripts=[],
        expected_outputs=output_paths
    )
    
    # Generate random existence status for each output
    existence = {}
    for output_path in output_paths:
        existence[output_path] = data.draw(st.booleans())
    
    # Set module and existence
    detail_view.set_module(module)
    detail_view.set_output_existence(existence)
    
    # Verify existence indicators are displayed correctly
    outputs_layout = detail_view.outputs_layout
    
    # Check each output path has correct indicator
    for output_path, exists in existence.items():
        # Find the indicator for this output
        found_indicator = False
        for i in range(outputs_layout.count()):
            item = outputs_layout.itemAt(i)
            if item.layout():
                # Check if this layout contains our output path
                path_found = False
                indicator_text = None
                
                for j in range(item.layout().count()):
                    widget = item.layout().itemAt(j).widget()
                    if widget and hasattr(widget, 'text'):
                        text = widget.text()
                        if output_path in text:
                            path_found = True
                        if text in ["✓", "✗"]:
                            indicator_text = text
                
                if path_found:
                    found_indicator = True
                    if exists:
                        assert indicator_text == "✓", (
                            f"Output '{output_path}' exists, should have ✓ indicator"
                        )
                    else:
                        assert indicator_text == "✗", (
                            f"Output '{output_path}' doesn't exist, should have ✗ indicator"
                        )
                    break
        
        assert found_indicator, f"Should find indicator for output '{output_path}'"


# Feature: pipeline-ui, Property 26: Missing Output Warning
@given(
    st.lists(st.text(min_size=1, max_size=50), min_size=2, max_size=10, unique=True)
)
def test_missing_output_warning_property(output_paths):
    """
    For any completed module with missing expected outputs, the UI should
    display a visual warning.
    
    Validates: Requirements 7.3
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    detail_view = ModuleDetailView()
    
    # Create module with expected outputs
    module = ModuleDefinition(
        id="M_TEST",
        name="Test Module",
        description="Test",
        scripts=[],
        expected_outputs=output_paths
    )
    
    # Set module
    detail_view.set_module(module)
    
    # Test case 1: All outputs exist - no warning
    existence_all = {path: True for path in output_paths}
    detail_view.set_output_existence(existence_all)
    
    # Check for warning label
    warning_found = False
    for i in range(detail_view.outputs_layout.count()):
        widget = detail_view.outputs_layout.itemAt(i).widget()
        if widget and hasattr(widget, 'text'):
            if "missing" in widget.text().lower():
                warning_found = True
                break
    
    assert not warning_found, "Should not show warning when all outputs exist"
    
    # Test case 2: Some outputs missing - warning should appear
    existence_partial = {path: (i % 2 == 0) for i, path in enumerate(output_paths)}
    if not all(existence_partial.values()):  # Ensure at least one is missing
        detail_view.set_output_existence(existence_partial)
        
        # Check for warning label
        warning_found = False
        for i in range(detail_view.outputs_layout.count()):
            widget = detail_view.outputs_layout.itemAt(i).widget()
            if widget and hasattr(widget, 'text'):
                text = widget.text().lower()
                if "missing" in text or "⚠" in widget.text():
                    warning_found = True
                    break
        
        assert warning_found, "Should show warning when some outputs are missing"


# Feature: pipeline-ui, Property 36: Script Parameters Display
@given(
    st.dictionaries(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')),
        st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(min_size=1, max_size=50)),
        min_size=1,
        max_size=10
    )
)
def test_script_parameters_display_property(parameters):
    """
    For any script with configuration parameters, the UI should display
    all parameter names, values, and descriptions.
    
    Validates: Requirements 12.1, 12.2
    """
    # Create app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    detail_view = ModuleDetailView()
    
    # Create module with script that has parameters
    module = ModuleDefinition(
        id="M_TEST",
        name="Test Module",
        description="Test",
        scripts=[
            ScriptInfo(
                name="test_script.py",
                path=Path("code/test_script.py"),
                description="Test script",
                parameters=parameters
            )
        ],
        expected_outputs=[]
    )
    
    # Set module
    detail_view.set_module(module)
    
    # Verify parameters are displayed
    scripts_layout = detail_view.scripts_layout
    
    # Get all text from script section
    displayed_text = []
    for i in range(scripts_layout.count()):
        widget = scripts_layout.itemAt(i).widget()
        if widget:
            # Get all child widgets recursively
            for child in widget.findChildren(type(widget).__bases__[0]):
                if hasattr(child, 'text'):
                    displayed_text.append(child.text())
    
    # Combine all text
    all_text = " ".join(displayed_text)
    
    # Verify each parameter name and value appears
    for param_name, param_value in parameters.items():
        assert param_name in all_text, (
            f"Parameter name '{param_name}' should be displayed"
        )
        assert str(param_value) in all_text, (
            f"Parameter value '{param_value}' should be displayed"
        )


# Unit tests

def test_module_title_display(module_detail_view, sample_module):
    """Test that module title is displayed correctly."""
    module_detail_view.set_module(sample_module)
    
    title_text = module_detail_view.module_title.text()
    assert sample_module.id in title_text
    assert sample_module.name in title_text


def test_module_description_display(module_detail_view, sample_module):
    """Test that module description is displayed."""
    module_detail_view.set_module(sample_module)
    
    desc_text = module_detail_view.module_description.text()
    assert sample_module.description in desc_text


def test_execute_button_present(module_detail_view, sample_module):
    """Test that execute button is present."""
    module_detail_view.set_module(sample_module)
    
    assert module_detail_view.execute_button is not None
    assert "Execute" in module_detail_view.execute_button.text()


def test_open_directory_button_present(module_detail_view, sample_module):
    """Test that open directory button is present."""
    module_detail_view.set_module(sample_module)
    
    assert module_detail_view.open_dir_button is not None
    assert "Directory" in module_detail_view.open_dir_button.text()


def test_scripts_section_present(module_detail_view, sample_module):
    """Test that scripts section is present."""
    module_detail_view.set_module(sample_module)
    
    assert module_detail_view.scripts_group is not None
    assert "Scripts" in module_detail_view.scripts_group.title()


def test_outputs_section_present(module_detail_view, sample_module):
    """Test that expected outputs section is present."""
    module_detail_view.set_module(sample_module)
    
    assert module_detail_view.outputs_group is not None
    assert "Output" in module_detail_view.outputs_group.title()


def test_script_display(module_detail_view, sample_module):
    """Test that scripts are displayed with details."""
    module_detail_view.set_module(sample_module)
    
    # Check scripts layout has content
    assert module_detail_view.scripts_layout.count() > 0


def test_parameter_categorization_display(module_detail_view, sample_module):
    """Test that parameter categorization is shown."""
    module_detail_view.set_module(sample_module)
    
    # Look for categorization text in scripts section
    found_categorization = False
    for i in range(module_detail_view.scripts_layout.count()):
        widget = module_detail_view.scripts_layout.itemAt(i).widget()
        if widget:
            for child in widget.findChildren(type(widget).__bases__[0]):
                if hasattr(child, 'text'):
                    text = child.text()
                    if "Common" in text and "DATE" in text and "MOUSE" in text and "RUN" in text:
                        found_categorization = True
                        break
    
    assert found_categorization, "Should display parameter categorization (Common vs Script-specific)"


def test_execution_enabled_control(module_detail_view, sample_module):
    """Test enabling/disabling execution button."""
    module_detail_view.set_module(sample_module)
    
    # Initially enabled
    module_detail_view.set_execution_enabled(True)
    assert module_detail_view.execute_button.isEnabled()
    
    # Disable
    module_detail_view.set_execution_enabled(False)
    assert not module_detail_view.execute_button.isEnabled()


def test_execute_signal(module_detail_view, sample_module, qtbot):
    """Test that execute_module signal is emitted."""
    module_detail_view.set_module(sample_module)
    
    with qtbot.waitSignal(module_detail_view.execute_module, timeout=1000) as blocker:
        module_detail_view.execute_button.click()
    
    assert blocker.args == [sample_module.id]


def test_open_directory_signal(module_detail_view, sample_module, qtbot):
    """Test that open_directory signal is emitted."""
    module_detail_view.set_module(sample_module)
    
    with qtbot.waitSignal(module_detail_view.open_directory, timeout=1000) as blocker:
        module_detail_view.open_dir_button.click()
    
    assert blocker.args == [sample_module.id]


def test_optional_module_indication(module_detail_view):
    """Test that optional modules are indicated."""
    optional_module = ModuleDefinition(
        id="M1.5",
        name="Optional Module",
        description="Test",
        scripts=[],
        expected_outputs=[],
        optional=True
    )
    
    module_detail_view.set_module(optional_module)
    
    title_text = module_detail_view.module_title.text()
    assert "Optional" in title_text


def test_interactive_module_indication(module_detail_view):
    """Test that interactive modules are indicated."""
    interactive_module = ModuleDefinition(
        id="M3",
        name="Interactive Module",
        description="Test",
        scripts=[],
        expected_outputs=[],
        requires_interaction=True
    )
    
    module_detail_view.set_module(interactive_module)
    
    title_text = module_detail_view.module_title.text()
    assert "Interactive" in title_text
