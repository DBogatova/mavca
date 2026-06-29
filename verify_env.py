"""Run this script to verify the mavca conda environment is set up correctly."""

import importlib
import sys

PACKAGES = [
    ("numpy",      "numpy"),
    ("cv2",        "opencv"),
    ("PyQt5",      "pyqt"),
    ("napari",     "napari"),
    ("scipy",      "scipy"),
    ("skimage",    "scikit-image"),
    ("matplotlib", "matplotlib"),
    ("pandas",     "pandas"),
    ("tifffile",   "tifffile"),
    ("imageio",    "imageio"),
    ("tqdm",       "tqdm"),
    ("numba",      "numba"),
    ("dask",       "dask"),
    ("vispy",      "vispy"),
]

passed = []
failed = []

for import_name, display_name in PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown version")
        passed.append(f"  {display_name:<16} {version}")
    except ImportError as e:
        failed.append(f"  {display_name:<16} MISSING  ({e})")

print(f"\nPython {sys.version}\n")
print(f"OK  ({len(passed)}/{len(PACKAGES)}):")
print("\n".join(passed))

if failed:
    print(f"\nFAILED ({len(failed)}/{len(PACKAGES)}):")
    print("\n".join(failed))
    print("\nEnvironment is incomplete. Re-run the installation steps for the missing packages.")
    sys.exit(1)
else:
    print("\nAll packages imported successfully. Environment is ready.")
