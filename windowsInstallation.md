# Windows Installation Guide

## Prerequisites

### 1. Install Anaconda

Anaconda is required to manage the Python environment and dependencies.

Download and install Anaconda from the official website:
 https://www.anaconda.com/download

Follow the installer prompts and accept the default settings. Once installed, open **Anaconda Prompt** from the Start Menu for all commands below.

---

## Setting Up the Environment

### 2. Create the Conda Environment

Create a new environment named `scape` with Python 3.11:

```bash
conda create --name scape python=3.11
```

Activate it:

```bash
conda activate scape
```

---

### 3. Install Dependencies from `requirementsWindows.txt`

Make sure you have navigated to the folder with requirementsWindows.txt or provide the full path.

```bash
pip install -r requirementsWindows.txt
```

---

### 5. Install Remaining Packages via Conda

After the pip install completes, install the following packages using conda:

```bash
conda install numpy=2.2.5
conda install opencv
conda install pyqt=5.15
```

> **Why conda for these?**
> - **PyQt5** must be installed via conda to avoid conflicts with Qt libraries. `conda install pyqt=5.15` automatically includes all required components (`PyQt5-Qt5`, `PyQt5_sip`, and the Qt5 binaries) — no need to install those separately.
> - **numpy** and **opencv** are installed via conda to ensure they link correctly against the conda environment's system libraries.

---

## Verify the Installation

To confirm everything is set up correctly, run:

```bash
python -c "import numpy; import cv2; import PyQt5; print('All packages imported successfully')"
```
