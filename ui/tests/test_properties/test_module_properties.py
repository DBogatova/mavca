"""
Property-based tests for ModuleDefinition and module display properties.

These tests use Hypothesis to verify universal correctness properties
across all possible module configurations.
"""

from hypothesis import given, strategies as st, settings
from pathlib import Path
from ui.models.module_definition import ModuleDefinition, ScriptInfo
from ui.models.pipeline_modules import MODULES, get_all_module_ids


# Feature: pipeline-ui, Property 1: Module Information Completeness
@given(st.sampled_from(get_all_module_ids()))
@settings(max_examples=100)
def test_module_information_completeness(module_id):
    """
    For any module definition, when rendered in the UI, the display should
    include the module number, name, and purpose.
    
    This property verifies that every module in the MODULES configuration
    has all three required pieces of information:
    - id (module number like "M1", "M1.5", etc.)
    - name (descriptive name)
    - description (purpose/description)
    
    Validates: Requirements 1.2
    """
    # Get the module definition
    module = MODULES[module_id]
    
    # Verify module has an id (module number)
    assert module.id is not None, f"Module {module_id} missing id"
    assert isinstance(module.id, str), f"Module {module_id} id is not a string"
    assert len(module.id) > 0, f"Module {module_id} has empty id"
    assert module.id == module_id, f"Module id mismatch: expected {module_id}, got {module.id}"
    
    # Verify module has a name
    assert module.name is not None, f"Module {module_id} missing name"
    assert isinstance(module.name, str), f"Module {module_id} name is not a string"
    assert len(module.name) > 0, f"Module {module_id} has empty name"
    
    # Verify module has a description (purpose)
    assert module.description is not None, f"Module {module_id} missing description"
    assert isinstance(module.description, str), f"Module {module_id} description is not a string"
    assert len(module.description) > 0, f"Module {module_id} has empty description"
    
    # Additional verification: all three fields should be distinct
    # (a module shouldn't have the same text for id, name, and description)
    assert not (module.id == module.name == module.description), (
        f"Module {module_id} has identical id, name, and description"
    )


# Feature: pipeline-ui, Property 1: Module Information Completeness (Extended)
@given(
    st.text(min_size=1, max_size=10),
    st.text(min_size=1, max_size=100),
    st.text(min_size=1, max_size=200),
    st.lists(
        st.builds(
            ScriptInfo,
            name=st.text(min_size=1, max_size=50),
            path=st.builds(Path, st.text(min_size=1, max_size=100)),
            description=st.text(min_size=1, max_size=200),
            optional=st.booleans()
        ),
        min_size=1,
        max_size=5
    ),
    st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10),
    st.booleans(),
    st.booleans()
)
@settings(max_examples=100)
def test_module_information_completeness_arbitrary_modules(
    module_id, name, description, scripts, expected_outputs, optional, requires_interaction
):
    """
    For any arbitrary module definition (not just the predefined MODULES),
    the module should have complete information including id, name, and description.
    
    This property verifies that the ModuleDefinition dataclass enforces
    the presence of all required fields for any possible module configuration.
    
    Validates: Requirements 1.2
    """
    # Create a module with arbitrary data
    module = ModuleDefinition(
        id=module_id,
        name=name,
        description=description,
        scripts=scripts,
        expected_outputs=expected_outputs,
        optional=optional,
        requires_interaction=requires_interaction
    )
    
    # Verify module has an id (module number)
    assert module.id is not None, "Module missing id"
    assert isinstance(module.id, str), "Module id is not a string"
    assert len(module.id) > 0, "Module has empty id"
    assert module.id == module_id, f"Module id mismatch: expected {module_id}, got {module.id}"
    
    # Verify module has a name
    assert module.name is not None, "Module missing name"
    assert isinstance(module.name, str), "Module name is not a string"
    assert len(module.name) > 0, "Module has empty name"
    assert module.name == name, f"Module name mismatch: expected {name}, got {module.name}"
    
    # Verify module has a description (purpose)
    assert module.description is not None, "Module missing description"
    assert isinstance(module.description, str), "Module description is not a string"
    assert len(module.description) > 0, "Module has empty description"
    assert module.description == description, f"Module description mismatch: expected {description}, got {module.description}"
    
    # Verify that all three required fields are present and accessible
    # (this simulates what a UI renderer would need to access)
    display_info = {
        'id': module.id,
        'name': module.name,
        'description': module.description
    }
    
    assert all(v is not None for v in display_info.values()), (
        "Not all display information is available"
    )
    assert all(isinstance(v, str) for v in display_info.values()), (
        "Not all display information is string type"
    )
    assert all(len(v) > 0 for v in display_info.values()), (
        "Some display information is empty"
    )


# Feature: pipeline-ui, Property 4: Script List Completeness
@given(st.sampled_from(get_all_module_ids()))
@settings(max_examples=100)
def test_script_list_completeness(module_id):
    """
    For any module with N scripts, the UI should display all N scripts.
    
    This property verifies that every module in the MODULES configuration
    exposes all of its scripts, and that the scripts list is complete and accessible.
    The UI must be able to display all scripts belonging to a module.
    
    Validates: Requirements 1.5
    """
    # Get the module definition
    module = MODULES[module_id]
    
    # Verify module has a scripts list
    assert module.scripts is not None, f"Module {module_id} has no scripts list"
    assert isinstance(module.scripts, list), f"Module {module_id} scripts is not a list"
    
    # Get the number of scripts
    n_scripts = len(module.scripts)
    
    # Verify that the module has at least one script
    # (every module should have at least one script to execute)
    assert n_scripts > 0, f"Module {module_id} has no scripts"
    
    # Verify that all N scripts are accessible and have required information
    for i, script in enumerate(module.scripts):
        assert script is not None, f"Module {module_id} script {i} is None"
        assert isinstance(script, ScriptInfo), (
            f"Module {module_id} script {i} is not a ScriptInfo instance"
        )
        
        # Each script must have a name (for display in UI)
        assert script.name is not None, f"Module {module_id} script {i} has no name"
        assert isinstance(script.name, str), f"Module {module_id} script {i} name is not a string"
        assert len(script.name) > 0, f"Module {module_id} script {i} has empty name"
        
        # Each script must have a path (for execution)
        assert script.path is not None, f"Module {module_id} script {i} has no path"
        assert isinstance(script.path, Path), f"Module {module_id} script {i} path is not a Path"
        
        # Each script must have a description (for display in UI)
        assert script.description is not None, f"Module {module_id} script {i} has no description"
        assert isinstance(script.description, str), (
            f"Module {module_id} script {i} description is not a string"
        )
        assert len(script.description) > 0, f"Module {module_id} script {i} has empty description"
    
    # Verify that the UI can iterate through all N scripts
    # (simulating what a UI renderer would do)
    displayed_scripts = []
    for script in module.scripts:
        displayed_scripts.append({
            'name': script.name,
            'description': script.description,
            'optional': script.optional
        })
    
    # The number of displayed scripts should equal N
    assert len(displayed_scripts) == n_scripts, (
        f"Module {module_id} should display {n_scripts} scripts, "
        f"but only {len(displayed_scripts)} were accessible"
    )
    
    # All script names should be unique within a module
    # (to avoid confusion in the UI)
    script_names = [script.name for script in module.scripts]
    assert len(script_names) == len(set(script_names)), (
        f"Module {module_id} has duplicate script names: {script_names}"
    )


# Feature: pipeline-ui, Property 4: Script List Completeness (Extended)
@given(
    st.text(min_size=1, max_size=10),
    st.lists(
        st.builds(
            ScriptInfo,
            name=st.text(min_size=1, max_size=50),
            path=st.builds(Path, st.text(min_size=1, max_size=100)),
            description=st.text(min_size=1, max_size=200),
            optional=st.booleans()
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=100)
def test_script_list_completeness_arbitrary_modules(module_id, scripts):
    """
    For any arbitrary module with N scripts, all N scripts should be
    accessible and displayable.
    
    This property verifies that the ModuleDefinition dataclass correctly
    stores and exposes all scripts, regardless of the number of scripts.
    
    Validates: Requirements 1.5
    """
    # Create a module with arbitrary scripts
    module = ModuleDefinition(
        id=module_id,
        name="Test Module",
        description="Test module for script list completeness",
        scripts=scripts,
        expected_outputs=[]
    )
    
    # Get the number of scripts
    n_scripts = len(scripts)
    
    # Verify that the module has the expected number of scripts
    assert len(module.scripts) == n_scripts, (
        f"Module should have {n_scripts} scripts, but has {len(module.scripts)}"
    )
    
    # Verify that all N scripts are accessible
    for i in range(n_scripts):
        script = module.scripts[i]
        assert script is not None, f"Script {i} is None"
        assert isinstance(script, ScriptInfo), f"Script {i} is not a ScriptInfo instance"
        
        # Verify script has required display information
        assert script.name is not None and len(script.name) > 0, (
            f"Script {i} has no name"
        )
        assert script.description is not None and len(script.description) > 0, (
            f"Script {i} has no description"
        )
        assert script.path is not None, f"Script {i} has no path"
    
    # Verify that iterating through scripts yields exactly N scripts
    script_count = 0
    for script in module.scripts:
        script_count += 1
        assert script is not None, "Encountered None script during iteration"
    
    assert script_count == n_scripts, (
        f"Expected to iterate through {n_scripts} scripts, "
        f"but iterated through {script_count}"
    )
    
    # Verify that the scripts list is not modified during access
    # (the UI should be able to access scripts multiple times)
    first_access = [s.name for s in module.scripts]
    second_access = [s.name for s in module.scripts]
    assert first_access == second_access, (
        "Scripts list changed between accesses"
    )
