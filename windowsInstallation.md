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

Create a new environment named `env_name` with Python 3.11:

```bash
conda create --name env_name python=3.11
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
---

## Verify the Installation

To confirm everything is set up correctly, run the verification script from the repo root:

```bash
python verify_env.py
```

This checks all key packages (numpy, opencv, PyQt5, napari, scipy, scikit-image, matplotlib, pandas, tifffile, imageio, tqdm, numba, dask, vispy) and prints each one's version. If any are missing, it lists them and exits with a non-zero code so you know exactly what to reinstall.
