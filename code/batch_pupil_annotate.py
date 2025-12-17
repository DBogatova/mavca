#!/usr/bin/env python
"""
Batch pupil annotation for multiple TIFF files
"""

from pathlib import Path
from pupil_annotator import PupilAnnotator
import sys

def batch_annotate(folder_path):
    folder = Path(folder_path)
    tiff_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    
    if not tiff_files:
        print(f"No TIFF files found in {folder}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    for i, tiff_file in enumerate(tiff_files):
        print(f"\nAnnotating {i+1}/{len(tiff_files)}: {tiff_file.name}")
        
        # Check if already annotated
        annotation_file = tiff_file.parent / f"{tiff_file.stem}_pupil_annotations.csv"
        if annotation_file.exists():
            response = input(f"Annotation exists for {tiff_file.name}. Overwrite? (y/n): ")
            if response.lower() != 'y':
                continue
        
        annotator = PupilAnnotator(tiff_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_pupil_annotate.py <folder_path>")
        sys.exit(1)
    
    batch_annotate(sys.argv[1])