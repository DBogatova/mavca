#!/usr/bin/env python
"""
Pupil annotation tool with automatic tracking
Manually annotate 10-20 frames, then auto-track the rest
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import pandas as pd
from scipy.optimize import minimize
from skimage import filters, measure
from sklearn.cluster import KMeans
import cv2

class PupilAnnotator:
    def __init__(self, tiff_path):
        self.tiff_path = Path(tiff_path)
        self.points = []
        self.point_labels = [
            'pupil-top', 'pupil-right-top', 'pupil-right-mid', 'pupil-right-bottom',
            'pupil-bottom', 'pupil-left-bottom', 'pupil-left-mid', 'pupil-left-top',
            'lid-top', 'lid-bottom'
        ]
        
        # Load all frames
        with tifffile.TiffFile(tiff_path) as tif:
            self.images = [page.asarray() for page in tif.pages]
        
        self.current_frame = 0
        self.annotations = {}  # frame_idx: [(x,y), ...]
        self.setup_gui()
    
    def setup_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.update_display()
        
        # Add buttons
        ax_prev = plt.axes([0.1, 0.01, 0.08, 0.05])
        ax_next = plt.axes([0.19, 0.01, 0.08, 0.05])
        ax_clear = plt.axes([0.3, 0.01, 0.08, 0.05])
        ax_track = plt.axes([0.6, 0.01, 0.15, 0.05])
        ax_save = plt.axes([0.76, 0.01, 0.08, 0.05])
        
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_track = Button(ax_track, 'Auto Track')
        self.btn_save = Button(ax_save, 'Save')
        
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_clear.on_clicked(self.clear_points)
        self.btn_track.on_clicked(self.auto_track)
        self.btn_save.on_clicked(self.save_all)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.show()
    
    def update_display(self):
        self.ax.clear()
        self.ax.imshow(self.images[self.current_frame], cmap='gray')
        
        # Show existing points for current frame
        if self.current_frame in self.annotations:
            points = self.annotations[self.current_frame]
            for i, (x, y) in enumerate(points):
                self.ax.plot(x, y, 'ro', markersize=8)
                self.ax.text(x+5, y-5, str(i+1), color='red', fontsize=12)
        
        n_points = len(self.annotations.get(self.current_frame, []))
        self.ax.set_title(f'Frame {self.current_frame+1}/{len(self.images)} - Points: {n_points}/10')
        self.fig.canvas.draw()
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        current_points = self.annotations.get(self.current_frame, [])
        if len(current_points) >= 10:
            return
        
        x, y = event.xdata, event.ydata
        current_points.append((x, y))
        self.annotations[self.current_frame] = current_points
        self.update_display()
    
    def prev_frame(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()
    
    def next_frame(self, event):
        if self.current_frame < len(self.images) - 1:
            self.current_frame += 1
            self.update_display()
    
    def clear_points(self, event):
        if self.current_frame in self.annotations:
            del self.annotations[self.current_frame]
        self.update_display()
    
    def auto_track(self, event):
        if len(self.annotations) < 3:
            print("Need at least 3 annotated frames for tracking")
            return
        
        print(f"Auto-tracking {len(self.images)} frames from {len(self.annotations)} manual annotations...")
        
        # Get reference pupil from first annotated frame
        ref_frame = min(self.annotations.keys())
        ref_points = np.array(self.annotations[ref_frame][:8])  # pupil points only
        ref_center = np.mean(ref_points, axis=0)
        ref_radius = np.mean(np.linalg.norm(ref_points - ref_center, axis=1))
        
        for frame_idx in range(len(self.images)):
            if frame_idx in self.annotations:
                continue
                
            # Simple template matching approach
            frame = self.images[frame_idx]
            
            # Find pupil using thresholding and contour detection
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find contour closest to reference size
                best_contour = max(contours, key=cv2.contourArea)
                
                # Fit ellipse and generate 8 points around pupil
                if len(best_contour) >= 5:
                    ellipse = cv2.fitEllipse(best_contour)
                    center, (w, h), angle = ellipse
                    
                    # Generate 8 points around ellipse
                    angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 points
                    pupil_points = []
                    for a in angles:
                        x = center[0] + (w/2) * np.cos(a + np.radians(angle))
                        y = center[1] + (h/2) * np.sin(a + np.radians(angle))
                        pupil_points.append((x, y))
                    
                    # Add eyelid points (estimate from pupil)
                    lid_top = (center[0], center[1] - h/2 - 20)
                    lid_bottom = (center[0], center[1] + h/2 + 20)
                    
                    self.annotations[frame_idx] = pupil_points + [lid_top, lid_bottom]
        
        print(f"Tracking complete! Annotated {len(self.annotations)} frames.")
        self.update_display()
    
    def save_all(self, event):
        if not self.annotations:
            print("No annotations to save")
            return
        
        # Create CSV with all frames
        bodyparts = []
        coords = []
        
        for label in self.point_labels:
            bodyparts.extend([label, label, label])
        coords.extend(['x', 'y', 'likelihood'] * 10)
        
        output_path = self.tiff_path.parent / f"{self.tiff_path.stem}_pupil_annotations.csv"
        
        with open(output_path, 'w') as f:
            f.write(','.join(['bodyparts'] + bodyparts) + '\n')
            f.write(','.join(['coords'] + coords) + '\n')
            
            for frame_idx in sorted(self.annotations.keys()):
                points = self.annotations[frame_idx]
                if len(points) == 10:
                    row_data = []
                    for x, y in points:
                        row_data.extend([x, y, 1.0])
                    f.write(','.join([f'frame_{frame_idx}'] + [str(x) for x in row_data]) + '\n')
        
        print(f"Saved {len(self.annotations)} frames to: {output_path}")
        plt.close(self.fig)

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pupil_annotator.py <path_to_tiff>")
        print("\nWorkflow:")
        print("1. Manually annotate 10-20 frames (use Prev/Next buttons)")
        print("2. Click 'Auto Track' to detect pupils in remaining frames")
        print("3. Click 'Save' to export all annotations")
        return
    
    tiff_path = sys.argv[1]
    if not Path(tiff_path).exists():
        print(f"File not found: {tiff_path}")
        return
    
    annotator = PupilAnnotator(tiff_path)

if __name__ == "__main__":
    main()