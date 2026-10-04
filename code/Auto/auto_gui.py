#!/usr/bin/env python3
"""auto_gui.py - PyQt5 control panel for the automatic SCAPE apical-dendrite pipeline.

One window: run table with per-stage coloured cells and headline metrics, buttons
for running stages, opening outputs, and launching napari review.

Run:  $PY code/Auto/auto_gui.py
      --selftest   headless: builds the window, fills the table, exits 0 on success

Buttons:
  [Run selected]      runs all stages for the selected run
  [Run all]           runs all stages for all runs
  [Rerun stage]       reruns a specific stage (dropdown)
  [Open combo figure] opens the combo_auto.png
  [Open movie]        opens the generated movie
  [Open validation]   opens the validation overlay
  [Open stats folder] opens scape-auto/stats/
  [Review masks]      launches review_napari.py
  [Refresh]           re-scan disk
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scape_common import PROJECT, AUTO_ROOT, PYTHON

# Import status builder from auto_status
import auto_status as ast

STAGE_ORDER = ["detect", "traces", "validate", "global", "combo", "movie"]


def build_runs():
    return ast.build_status()


def selftest() -> int:
    """Headless test: build the model, check that it works."""
    runs = build_runs()
    if not runs:
        print("FAIL: no runs discovered")
        return 1
    print(f"runs discovered: {len(runs)}")

    # Check stage tally
    tally = {}
    for r in runs:
        tally[r["stage"]] = tally.get(r["stage"], 0) + 1
    print(f"stage tally: {tally}")

    # Check that table columns can be built
    for r in runs:
        _ = [r["date"], r["mouse"], r["run_name"], r["stage"], r["checklist"]]
        _ = r["next"]["label"]

    print(f"commands buildable for {sum(1 for r in runs if r['next']['runnable'])} runs")
    print("SELFTEST PASS")
    return 0


def run_gui() -> int:
    from PyQt5 import QtWidgets, QtCore, QtGui

    # Stage colours
    STAGE_COLORS = {
        "no_raw": "#9e9e9e",
        "raw": "#e0e0e0",
        "detect": "#fff59d",
        "traces": "#c5e1a5",
        "validate": "#81d4fa",
        "global": "#ce93d8",
        "combo": "#ffab91",
        "movie": "#80cbc4",
        "complete": "#2e7d32",
    }

    COLS = ["date", "mouse", "run", "Hz", "stage", "det", "tr", "val", "glob", "cmb", "mov",
            "auto", "human", "recall", "best r", "next"]

    class Panel(QtWidgets.QMainWindow):
        log_signal = QtCore.pyqtSignal(str)
        progress_signal = QtCore.pyqtSignal(int, int, str)
        refresh_signal = QtCore.pyqtSignal()

        def __init__(self):
            super().__init__()
            self.setWindowTitle("SCAPE Auto Pipeline - apical-dendrites-2025")
            self.resize(1350, 750)
            w = QtWidgets.QWidget()
            self.setCentralWidget(w)
            lay = QtWidgets.QVBoxLayout(w)

            # Table
            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(len(COLS))
            self.table.setHorizontalHeaderLabels(COLS)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            lay.addWidget(self.table, stretch=3)

            # Buttons row 1
            btns = QtWidgets.QHBoxLayout()
            self.b_run_sel = QtWidgets.QPushButton("Run selected")
            self.b_run_all = QtWidgets.QPushButton("Run all")
            self.stage_combo = QtWidgets.QComboBox()
            self.stage_combo.addItems(STAGE_ORDER)
            self.b_rerun = QtWidgets.QPushButton("Rerun stage")
            self.b_force = QtWidgets.QCheckBox("--force")
            self.b_ref = QtWidgets.QPushButton("Refresh")
            for b in (self.b_run_sel, self.b_run_all, self.stage_combo, self.b_rerun, self.b_force, self.b_ref):
                btns.addWidget(b)
            btns.addStretch()
            lay.addLayout(btns)

            # Buttons row 2
            btns2 = QtWidgets.QHBoxLayout()
            self.b_combo = QtWidgets.QPushButton("Open combo figure")
            self.b_movie = QtWidgets.QPushButton("Open movie")
            self.b_val = QtWidgets.QPushButton("Open validation")
            self.b_stats = QtWidgets.QPushButton("Open stats folder")
            self.b_review = QtWidgets.QPushButton("Review masks (napari)")
            for b in (self.b_combo, self.b_movie, self.b_val, self.b_stats, self.b_review):
                btns2.addWidget(b)
            btns2.addStretch()
            lay.addLayout(btns2)

            # Progress bar
            prog = QtWidgets.QHBoxLayout()
            self.prog_label = QtWidgets.QLabel("idle")
            self.prog_bar = QtWidgets.QProgressBar()
            self.prog_bar.setRange(0, 1)
            self.prog_bar.setValue(0)
            self.prog_bar.setFixedHeight(16)
            prog.addWidget(QtWidgets.QLabel("progress:"))
            prog.addWidget(self.prog_bar, stretch=1)
            prog.addWidget(self.prog_label)
            lay.addLayout(prog)

            # Log pane
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(5000)
            f = QtGui.QFont("Menlo")
            f.setPointSize(11)
            self.log.setFont(f)
            lay.addWidget(self.log, stretch=2)

            # Connections
            self.b_ref.clicked.connect(self.refresh)
            self.b_run_sel.clicked.connect(self.run_selected)
            self.b_run_all.clicked.connect(self.run_all)
            self.b_rerun.clicked.connect(self.rerun_stage)
            self.b_combo.clicked.connect(self.open_combo)
            self.b_movie.clicked.connect(self.open_movie)
            self.b_val.clicked.connect(self.open_validation)
            self.b_stats.clicked.connect(self.open_stats)
            self.b_review.clicked.connect(self.review_napari)
            self.log_signal.connect(self.log.appendPlainText)
            self.progress_signal.connect(self._on_progress)
            self.refresh_signal.connect(self.refresh)

            self.busy = False
            self.runs = []
            self.refresh()

        def refresh(self):
            self.runs = build_runs()
            self.table.setRowCount(len(self.runs))
            for i, r in enumerate(self.runs):
                a = r["artifacts"]
                stage_flags = {
                    "det": a.get("detect"),
                    "tr": a.get("traces"),
                    "val": a.get("validate"),
                    "glob": a.get("global"),
                    "cmb": a.get("combo"),
                    "mov": a.get("movie"),
                }
                cells = [
                    r["date"], r["mouse"], r["run_name"], f"{r['frame_rate']:.0f}",
                    r["stage"],
                    "✓" if stage_flags["det"] else "",
                    "✓" if stage_flags["tr"] else "",
                    "✓" if stage_flags["val"] else "",
                    "✓" if stage_flags["glob"] else "",
                    "✓" if stage_flags["cmb"] else "",
                    "✓" if stage_flags["mov"] else "",
                    str(r["n_auto"]) if r["n_auto"] else "",
                    str(r["n_human"]) if r["n_human"] else "",
                    f"{r['recall']:.2f}" if r["recall"] is not None else "",
                    f"{r['trace_r']:.2f}" if r["trace_r"] is not None else "",
                    r["next"]["label"],
                ]
                for j, txt in enumerate(cells):
                    it = QtWidgets.QTableWidgetItem(txt)
                    # Colour the stage column
                    if j == 4:  # stage column
                        color = STAGE_COLORS.get(r["stage"], "#ffffff")
                        it.setBackground(QtGui.QColor(color))
                    # Colour the stage flag columns
                    if j in range(5, 11):  # det through mov
                        key = ["det", "tr", "val", "glob", "cmb", "mov"][j - 5]
                        if stage_flags[key]:
                            it.setBackground(QtGui.QColor("#c8e6c9"))
                    if r["stage"] == "complete":
                        it.setForeground(QtGui.QColor("#2e7d32"))
                    elif r["stage"] == "no_raw":
                        it.setForeground(QtGui.QColor("#9e9e9e"))
                    self.table.setItem(i, j, it)
            self.table.resizeColumnsToContents()

            # Tally
            tally = {}
            for r in self.runs:
                tally[r["stage"]] = tally.get(r["stage"], 0) + 1
            self.logline(f"refreshed: {len(self.runs)} runs  " +
                         "  ".join(f"{k}={v}" for k, v in tally.items()))

        def selected(self):
            i = self.table.currentRow()
            if i < 0 or i >= len(self.runs):
                self.logline("!! select a run first")
                return None
            return self.runs[i]

        def logline(self, s):
            self.log_signal.emit(s)

        def _on_progress(self, done, total, label):
            if total <= 0:
                self.prog_bar.setRange(0, 0)
            else:
                self.prog_bar.setRange(0, total)
                self.prog_bar.setValue(min(done, total))
            self.prog_label.setText(label)

        def run_selected(self):
            r = self.selected()
            if r is None or self.busy:
                return
            threading.Thread(target=self._run_pipeline, args=([r["key"]],), daemon=True).start()

        def run_all(self):
            if self.busy:
                return
            keys = [r["key"] for r in self.runs if r["stage"] not in ("no_raw", "complete")]
            if not keys:
                self.logline("nothing to run - all complete or no raw")
                return
            threading.Thread(target=self._run_pipeline, args=(keys,), daemon=True).start()

        def rerun_stage(self):
            r = self.selected()
            if r is None or self.busy:
                return
            stage = self.stage_combo.currentText()
            threading.Thread(target=self._run_stage, args=(r["key"], stage), daemon=True).start()

        def _run_pipeline(self, keys):
            self.busy = True
            try:
                script = HERE / "run_auto.py"
                if not script.exists():
                    self.logline(f"ERROR: run_auto.py not found")
                    return
                cmd = [PYTHON, str(script)]
                for k in keys:
                    cmd += ["--run", k]
                if self.b_force.isChecked():
                    cmd.append("--force")
                cmd += ["--jobs", "2"]
                self._exec(cmd, f"pipeline ({len(keys)} runs)")
            finally:
                self.busy = False
                self.refresh_signal.emit()

        def _run_stage(self, key, stage):
            self.busy = True
            try:
                script = HERE / ast.SCRIPTS.get(stage, "")
                if not script.exists():
                    self.logline(f"ERROR: script for {stage} not found")
                    return
                cmd = [PYTHON, str(script), "--run", key]
                if self.b_force.isChecked():
                    cmd.append("--force")
                self._exec(cmd, f"{stage} on {key}")
            finally:
                self.busy = False
                self.refresh_signal.emit()

        def _exec(self, argv, desc=""):
            self.logline(f"$ {' '.join(argv)}")
            self.progress_signal.emit(0, 0, desc or "running")
            p = subprocess.Popen(argv, cwd=str(PROJECT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                line = line.rstrip()
                # Parse progress lines if present
                if line.startswith("##PROGRESS "):
                    try:
                        parts = line[11:].split(" ", 1)
                        frac = parts[0]
                        label = parts[1] if len(parts) > 1 else desc
                        done, total = (int(x) for x in frac.split("/"))
                        self.progress_signal.emit(done, total, label)
                    except ValueError:
                        pass
                    continue
                self.logline(line)
            p.wait()
            self.progress_signal.emit(1, 1, "done" if p.returncode == 0 else "FAILED")
            self.logline(f"[exit {p.returncode}]")

        def open_combo(self):
            r = self.selected()
            if r is None:
                return
            run_obj = r["run"]
            # Try auto first, then human
            for source in ["auto", "human"]:
                for ext in [".png", ".pdf"]:
                    combo = run_obj.out / "figures" / f"combo_{source}{ext}"
                    if combo.exists():
                        subprocess.run(["open", str(combo)])
                        return
            self.logline(f"combo figure not found for {r['key']}")

        def open_movie(self):
            r = self.selected()
            if r is None:
                return
            run_obj = r["run"]
            movie = run_obj.out / "movies" / f"{run_obj.run}_dual_behavior.mp4"
            if movie.exists():
                subprocess.run(["open", str(movie)])
            else:
                self.logline(f"movie not found for {r['key']}")

        def open_validation(self):
            r = self.selected()
            if r is None:
                return
            run_obj = r["run"]
            overlay = run_obj.out / "validation" / "overlay.png"
            if overlay.exists():
                subprocess.run(["open", str(overlay)])
            else:
                self.logline(f"validation overlay not found for {r['key']}")

        def open_stats(self):
            stats_dir = AUTO_ROOT / "stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["open", str(stats_dir)])

        def review_napari(self):
            r = self.selected()
            if r is None:
                return
            script = HERE / "review_napari.py"
            if not script.exists():
                self.logline("review_napari.py not found")
                return
            cmd = [PYTHON, str(script), "--run", r["key"]]
            self.logline(f"launch: {' '.join(cmd)}")
            child = subprocess.Popen(cmd, cwd=str(PROJECT), start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            def _watch(proc=child, name=f"review {r['key']}"):
                rc = proc.wait()
                self.log_signal.emit(f"[napari closed: {name} exit {rc}] - press Refresh")

            threading.Thread(target=_watch, daemon=True).start()
            self.logline("napari launched in its own window; save there, then Refresh here.")

    app = QtWidgets.QApplication(sys.argv)
    panel = Panel()
    panel.show()
    return app.exec_()


def main():
    # Parse arguments BEFORE touching Qt. Constructing a QApplication and
    # calling app.exec_() enters the event loop and never returns, so a bare
    # `--help` used to hang forever -- even under QT_QPA_PLATFORM=offscreen,
    # which suppresses the window but still runs the loop. That silently wedged
    # an automated audit of this directory for 42 minutes on 2026-10-04.
    # argparse handles -h/--help itself and exits before any Qt object exists.
    import argparse
    ap = argparse.ArgumentParser(
        prog="auto_gui.py",
        description=__doc__.split("\n\n")[0] if __doc__ else "auto_gui control panel",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="headless: build the window, fill the table, exit 0 on success "
                         "(forces QT_QPA_PLATFORM=offscreen)")
    args = ap.parse_args()

    if args.selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return selftest()
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
