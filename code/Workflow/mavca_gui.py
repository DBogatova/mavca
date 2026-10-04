#!/usr/bin/env python
"""mavca_gui.py - button-driven control panel for the MAVCA (SCAPE dendrite) pipeline.

The apical-dendrites-2025 counterpart of femtonics-data/code/STEP7_workflow/femto_gui.py.
Same shape: one window, a run table with live disk-driven stage detection, buttons that
run the next step for the selected run, a log pane that shows every command verbatim
so anything the panel does can be reproduced in a terminal.

  [Run next automatic step]  M1 / M1.5 / M2 / M2b / M4 / M5 / behaviour, executed as a
                             subprocess with live log output; chains until the run needs
                             a human (M3 napari) or is complete (checkbox controls chaining).
  [Open M3 curation (napari)] launches filter_selected_masks_m3.py detached, so napari's
                             event loop never fights this panel's.
  [Refresh]                  re-scan the disk.

WHAT IS MAVCA-SPECIFIC
----------------------
* Per-run params column (Hz / skip / mask source). Every command passes them explicitly
  through run_stage.py, so the 6 Hz sessions and the 14 s session never run at defaults.
  Runs whose session is NOT in mavca_status.SESSION_PARAMS are shown in orange and their
  auto step is refused until you add them - this is the single most expensive mistake the
  steering file documents, so the panel makes it impossible rather than merely visible.
* Shared-FOV runs skip M2/M2b/M3 and go straight to M4 with the source run's masks.
* A "STALE-OWN-MASKS" flag when a shared-FOV run still carries an old local mask set that
  differs from the source run's (0416 run2/run3: 80 local vs 54 used) - a reminder that
  those folders are dead weight, not a source of truth.

Run:  $PY code/Workflow/mavca_gui.py
      --selftest   headless: builds the table model + every command, checks that every
                   referenced script exists and every param is explicit, prints PASS/FAIL.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mavca_status as ms  # single source of truth for stages/params/commands

ROOT = ms.PROJECT_ROOT


def build_runs():
    return ms.build_status()


# ---------------------------------------------------------------------------
def selftest() -> int:
    runs = build_runs()
    assert runs, "no runs discovered under scape-data/"
    n_cmd = n_gui = 0
    problems = []
    for r in runs:
        nx = r["next"]
        cmd = nx["cmd"]
        if not cmd:
            continue
        # cmd[0]=python cmd[1]=run_stage.py cmd[2]=script
        for p in (cmd[0], cmd[1], cmd[2]):
            if not Path(p).exists():
                problems.append(f"{r['label']}: missing {p}")
        joined = " ".join(cmd)
        for must in ("--date", "--mouse", "--run", "SKIP_FIRST_SECONDS=", "FRAME_RATE="):
            if must not in joined:
                problems.append(f"{r['label']}: command lacks {must}")
        if r["mask_source"] and "save_traces_m4" in joined and f"MASK_SOURCE_RUN={r['mask_source']!r}" not in joined:
            problems.append(f"{r['label']}: shared-FOV M4 without MASK_SOURCE_RUN")
        if r["raw_prefix"] and "--raw-prefix" not in joined:
            problems.append(f"{r['label']}: raw prefix known but not passed")
        n_gui += nx["gui"]
        n_cmd += 1
    tally = {}
    for r in runs:
        tally[r["stage"]] = tally.get(r["stage"], 0) + 1
    print(f"runs discovered: {len(runs)}")
    print(f"stage tally: {tally}")
    print(f"commands built: {n_cmd}  (gui steps: {n_gui})")
    print(f"runs on default params: {sum(1 for r in runs if not r['params_known'])}")
    print(f"stale-own-mask flags: {sum(1 for r in runs if r['stale_own_masks'])}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        print("SELFTEST FAIL")
        return 1
    print("SELFTEST PASS")
    return 0


# ---------------------------------------------------------------------------
def run_gui() -> int:
    from qtpy import QtWidgets, QtCore, QtGui

    COLS = ["date", "mouse", "run", "Hz", "skip s", "masks", "stage", "flags", "next step"]

    class Panel(QtWidgets.QMainWindow):
        log_signal = QtCore.Signal(str)
        refresh_signal = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.setWindowTitle("MAVCA dendrite pipeline - apical-dendrites-2025")
            self.resize(1280, 700)
            w = QtWidgets.QWidget(); self.setCentralWidget(w)
            lay = QtWidgets.QVBoxLayout(w)

            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(len(COLS))
            self.table.setHorizontalHeaderLabels(COLS)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            lay.addWidget(self.table, stretch=3)

            btns = QtWidgets.QHBoxLayout()
            self.b_auto = QtWidgets.QPushButton("Run next automatic step")
            self.b_gui = QtWidgets.QPushButton("Open M3 curation (napari)")
            self.b_ref = QtWidgets.QPushButton("Refresh")
            self.chain = QtWidgets.QCheckBox("chain automatic steps")
            self.chain.setChecked(True)
            for b in (self.b_auto, self.b_gui, self.b_ref):
                btns.addWidget(b)
            btns.addWidget(self.chain)
            btns.addStretch()
            self.lbl = QtWidgets.QLabel("")
            btns.addWidget(self.lbl)
            lay.addLayout(btns)

            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(5000)
            f = QtGui.QFont("Menlo"); f.setPointSize(11)
            self.log.setFont(f)
            lay.addWidget(self.log, stretch=2)

            self.b_ref.clicked.connect(self.refresh)
            self.b_auto.clicked.connect(lambda: self.dispatch(gui_ok=False))
            self.b_gui.clicked.connect(lambda: self.dispatch(gui_ok=True))
            self.log_signal.connect(self.log.appendPlainText)
            self.refresh_signal.connect(self.refresh)
            self.busy = False
            self.runs = []
            self.refresh()

        # ---- table ------------------------------------------------------
        def refresh(self):
            self.runs = build_runs()
            self.table.setRowCount(len(self.runs))
            for i, r in enumerate(self.runs):
                nx = r["next"]
                cells = [r["date"], r["mouse"], r["run"], f"{r['frame_rate']:.0f}", f"{r['skip_s']:.0f}",
                         str(r["n_masks_effective"]), r["stage"], r["flags"],
                         ("[napari] " if nx["gui"] else "") + nx["label"]]
                for j, txt in enumerate(cells):
                    it = QtWidgets.QTableWidgetItem(txt)
                    if r["stage"] == "complete":
                        it.setForeground(QtGui.QColor("#2e7d32"))
                    elif r["stage"] == "no_raw":
                        it.setForeground(QtGui.QColor("#9e9e9e"))
                    elif not r["params_known"]:
                        it.setForeground(QtGui.QColor("#e65100"))
                    if r["stale_own_masks"] and j == 7:
                        it.setForeground(QtGui.QColor("#b71c1c"))
                    self.table.setItem(i, j, it)
            self.table.resizeColumnsToContents()
            tally = {}
            for r in self.runs:
                tally[r["stage"]] = tally.get(r["stage"], 0) + 1
            self.lbl.setText("  ".join(f"{k}={v}" for k, v in tally.items()))
            self.logline("table refreshed from disk")

        def selected(self):
            i = self.table.currentRow()
            if i < 0 or i >= len(self.runs):
                self.logline("!! select a run first")
                return None
            return self.runs[i]

        def logline(self, s):
            self.log_signal.emit(s)

        # ---- actions ----------------------------------------------------
        def dispatch(self, gui_ok: bool):
            r = self.selected()
            if r is None or self.busy:
                return
            nx = r["next"]
            if not nx["cmd"]:
                self.logline(f"[{r['label']}] {nx['label']}")
                return
            if not r["params_known"]:
                self.logline(f"[{r['label']}] REFUSED: session {r['date']}/{r['mouse']} is not in "
                             f"mavca_status.SESSION_PARAMS, so frame rate / skip would be guessed. "
                             f"Add it (see .kiro/steering/pipeline-context.md) and Refresh.")
                return
            if nx["gui"] and not gui_ok:
                self.logline(f"[{r['label']}] next step is napari ({nx['label']}) -> use 'Open M3 curation'")
                return
            if nx["gui"]:
                self.logline("launch: " + " ".join(nx["cmd"]))
                child = subprocess.Popen(nx["cmd"], cwd=str(ROOT), start_new_session=True,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                def _watch(proc=child, name=nx["label"]):
                    rc = proc.wait()
                    self.log_signal.emit(f"[napari closed: {name} exit {rc}] - press Refresh")
                threading.Thread(target=_watch, daemon=True).start()
                self.logline("napari launched in its own window; save there, then Refresh here.")
                return
            threading.Thread(target=self._run_auto, args=(r,), daemon=True).start()

        def _run_auto(self, r):
            self.busy = True
            try:
                for _ in range(8):
                    fresh = [x for x in build_runs() if x["label"] == r["label"]]
                    if not fresh:
                        break
                    nx = fresh[0]["next"]
                    if not nx["cmd"] or nx["gui"]:
                        self.logline(f"[{r['label']}] stopping: {nx['label']}")
                        break
                    if not self._exec(nx["cmd"]):
                        break
                    if not self.chain.isChecked():
                        break
            finally:
                self.busy = False
                self.refresh_signal.emit()

        def _exec(self, argv) -> bool:
            self.logline("$ " + " ".join(argv))
            p = subprocess.Popen(argv, cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                self.logline(line.rstrip())
            p.wait()
            self.logline(f"[exit {p.returncode}]")
            return p.returncode == 0

    app = QtWidgets.QApplication(sys.argv)
    panel = Panel()
    panel.show()
    return app.exec_()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run_gui())
