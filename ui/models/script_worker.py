"""
Background worker for script execution.

Runs pipeline scripts in a QThread so the UI stays responsive
and console output / progress bars update in real-time.
"""

from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal

from .pipeline_config import PipelineConfig
from .script_executor import ScriptExecutor


class ScriptWorker(QThread):
    """Executes a pipeline module in a background thread.

    Signals:
        output_line: Emitted for each line of script output (str)
        finished_ok: Emitted on successful completion with the module_id (str)
        finished_err: Emitted on failure with (module_id, error_message)
    """

    output_line = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str, str)

    def __init__(self, controller, module_id: str, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._module_id = module_id

    def run(self):
        """Execute the module in the background thread."""
        try:
            self._controller.execute_module(
                self._module_id,
                lambda line: self.output_line.emit(line),
            )
            self.finished_ok.emit(self._module_id)
        except Exception as exc:
            self.finished_err.emit(self._module_id, str(exc))
