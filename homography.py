"""
Homography Module for Smart Traffic Light System
Simple projection from side view to top view.
"""

import cv2
import numpy as np


class HomographyProjector:
    """Simple homography projection for traffic objects."""
    
    def __init__(self):
        """Initialize with default settings."""
        self.homography_matrix = None
        self.topview_size = (800, 600)  # Height, Width for top-view visualization (made bigger)
        
    def project_detections_to_topview(self, detections):
        """
        Simple projection: just use bottom center of bounding boxes.
        In a real implementation, you would apply homography transformation here.
        """
        projected_detections = {
            'positions': [],
            'class_names': detections['class_names'],
            'confidences': detections['confidences']
        }
        
        # For each detection, get the bottom center point (where object touches ground)
        for box in detections['boxes']:
            x1, y1, x2, y2 = box
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2  # Bottom of bounding box
            
            # TODO: Apply real homography transformation here
            # For now, just use the bottom center coordinates
            projected_detections['positions'].append((bottom_center_x, bottom_center_y))
        
        return projected_detections
    
    def create_topview_visualization(self, projected_detections, frame_width=1872, frame_height=998):
        """
        Create a simple top-down view visualization showing projected object positions.
        
        Args:
            projected_detections (dict): Projected detection results
            frame_width (int): Original video frame width
            frame_height (int): Original video frame height
            
        Returns:
            np.ndarray: Top-view visualization image
        """
        # Create blank canvas for top view
        topview_img = np.zeros((self.topview_size[0], self.topview_size[1], 3), dtype=np.uint8)
        
        # Draw a simple road layout (placeholder)
        # Draw road as gray rectangle
        road_margin = 50
        cv2.rectangle(topview_img, 
                     (road_margin, road_margin), 
                     (self.topview_size[1] - road_margin, self.topview_size[0] - road_margin),
                     (80, 80, 80), -1)  # Gray road
        
        # Draw center line
        center_x = self.topview_size[1] // 2
        cv2.line(topview_img, 
                (center_x, road_margin), 
                (center_x, self.topview_size[0] - road_margin),
                (255, 255, 255), 2)  # White center line
        
        # Color mapping for different object types
        colors = {
            'person': (0, 255, 0),      # Green for pedestrians
            'bicycle': (255, 0, 0),     # Blue for bicycles  
            'car': (0, 0, 255),         # Red for cars
            'motorcycle': (255, 0, 255), # Magenta for motorcycles
            'bus': (0, 255, 255),       # Yellow for buses
            'truck': (128, 0, 128),     # Purple for trucks
        }
        
        # Plot each detected object
        for i, (x, y) in enumerate(projected_detections['positions']):
            class_name = projected_detections['class_names'][i]
            color = colors.get(class_name, (255, 255, 255))
            
            # Convert original frame coordinates to top-view canvas coordinates
            # Simple scaling - in real implementation this would use homography
            canvas_x = int((x / frame_width) * (self.topview_size[1] - 2 * road_margin)) + road_margin
            canvas_y = int((y / frame_height) * (self.topview_size[0] - 2 * road_margin)) + road_margin
            
            # Ensure coordinates are within bounds
            canvas_x = max(road_margin, min(self.topview_size[1] - road_margin, canvas_x))
            canvas_y = max(road_margin, min(self.topview_size[0] - road_margin, canvas_y))
            
            # Draw object as circle (bigger for better visibility)
            cv2.circle(topview_img, (canvas_x, canvas_y), 12, color, -1)
            
            # Add label with bigger text
            cv2.putText(topview_img, class_name[:3], 
                       (canvas_x + 15, canvas_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add title with bigger text
        cv2.putText(topview_img, "Top View (Projected)", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add object count with bigger text
        total_objects = len(projected_detections['positions'])
        cv2.putText(topview_img, f"Objects: {total_objects}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return topview_img
