"""
Homography Module for Smart Traffic Light System
Interactive zebra crossing selection and homography transformation.
"""

import cv2
import numpy as np
import json
import os


class HomographyProjector:
    """Homography projection with interactive zebra crossing definition."""
    
    def __init__(self, config_file="zebra_config.json"):
        """Initialize with configuration file for zebra crossing points."""
        self.homography_matrix = None
        self.topview_size = (800, 600)  # Height, Width for top-view visualization
        self.config_file = config_file
        self.zebra_points = []  # Points defining the zebra crossing quadrilateral
        self.road_points = []   # Additional road reference points for better alignment
        self.zebra_configured = False
        
        # Real-time adjustment variables
        self.show_reference_lines = True
        self.dragging = False
        self.drag_point_index = -1
        self.drag_point_type = ""  # "zebra" or "road"
        self.drag_threshold = 15  # pixels
        self.adjustment_mode = False  # Toggle for enabling/disabling adjustments
        
        # Safety zone configuration
        self.safety_zone_configured = False
        self.safety_zone_top = []    # Top boundary line (2 points)
        self.safety_zone_bottom = []  # Bottom boundary line (2 points)
        self.show_safety_zones = True
        self.safety_zone_distance = 50  # Default distance from zebra crossing
        
        # Define top-view zebra crossing coordinates (destination) - centered
        canvas_center_x = self.topview_size[1] // 2  # 400
        canvas_center_y = self.topview_size[0] // 2  # 300
        zebra_width = 200
        zebra_height = 60
        
        self.topview_zebra_rect = np.array([
            [canvas_center_x - zebra_width//2, canvas_center_y - zebra_height//2],  # Top-left
            [canvas_center_x + zebra_width//2, canvas_center_y - zebra_height//2],  # Top-right  
            [canvas_center_x + zebra_width//2, canvas_center_y + zebra_height//2],  # Bottom-right
            [canvas_center_x - zebra_width//2, canvas_center_y + zebra_height//2]   # Bottom-left
        ], dtype=np.float32)
        
        # Road reference points mapped to full canvas corners
        self.topview_road_refs = np.array([
            [0, 0],                                    # Left road edge, far → top-left corner
            [self.topview_size[1], 0],                # Right road edge, far → top-right corner
            [0, self.topview_size[0]],                # Left road edge, near → bottom-left corner
            [self.topview_size[1], self.topview_size[0]]  # Right road edge, near → bottom-right corner
        ], dtype=np.float32)
        
        # Load existing configuration after all attributes are initialized
        self.load_zebra_config()
    
    def save_zebra_config(self):
        """Save zebra crossing, road reference, and safety zone configuration to file."""
        config = {
            'zebra_points': self.zebra_points,
            'road_points': self.road_points,
            'zebra_configured': self.zebra_configured,
            'safety_zone_top': self.safety_zone_top,
            'safety_zone_bottom': self.safety_zone_bottom,
            'safety_zone_configured': self.safety_zone_configured
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {self.config_file}")
    
    def load_zebra_config(self):
        """Load zebra crossing, road reference, and safety zone configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.zebra_points = config.get('zebra_points', [])
                self.road_points = config.get('road_points', [])
                self.zebra_configured = config.get('zebra_configured', False)
                self.safety_zone_top = config.get('safety_zone_top', [])
                self.safety_zone_bottom = config.get('safety_zone_bottom', [])
                self.safety_zone_configured = config.get('safety_zone_configured', False)
                if self.zebra_configured and len(self.zebra_points) == 4 and len(self.road_points) == 4:
                    self.compute_homography_enhanced()
                    print(f"Loaded configuration from {self.config_file}")
                elif self.zebra_configured and len(self.zebra_points) == 4:
                    self.compute_homography()
                    print(f"Loaded basic configuration from {self.config_file}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
    
    def setup_zebra_crossing_interactive(self, sample_frame):
        """Interactive setup of zebra crossing and road reference points."""
        print("\nEnhanced Road Alignment Setup")
        print("=" * 50)
        print("STEP 1: Click on the 4 corners of the zebra crossing:")
        print("1. Top-left corner")
        print("2. Top-right corner") 
        print("3. Bottom-right corner")
        print("4. Bottom-left corner")
        print("\nSTEP 2: Click on 4 road reference points:")
        print("5. Left road edge (far from camera)")
        print("6. Right road edge (far from camera)")
        print("7. Left road edge (near camera)")
        print("8. Right road edge (near camera)")
        print("\nPress 'r' to reset points, 's' to save, 'q' to quit")
        
        self.zebra_points = []
        self.road_points = []
        self.current_frame = sample_frame.copy()
        self.setup_step = "zebra"  # Track current setup step
        
        # Set up mouse callback
        cv2.namedWindow('Road Alignment Setup', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Road Alignment Setup', self.mouse_callback_enhanced)
        
        while True:
            display_frame = self.current_frame.copy()
            
            # Draw zebra crossing points (green)
            for i, point in enumerate(self.zebra_points):
                cv2.circle(display_frame, tuple(map(int, point)), 8, (0, 255, 0), -1)
                cv2.putText(display_frame, f"Z{i+1}", 
                           (int(point[0]+15), int(point[1]-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw road reference points (blue)
            for i, point in enumerate(self.road_points):
                cv2.circle(display_frame, tuple(map(int, point)), 8, (255, 0, 0), -1)
                cv2.putText(display_frame, f"R{i+1}", 
                           (int(point[0]+15), int(point[1]-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Draw lines for zebra crossing
            if len(self.zebra_points) > 1:
                for i in range(len(self.zebra_points)):
                    start = tuple(map(int, self.zebra_points[i]))
                    end = tuple(map(int, self.zebra_points[(i+1) % len(self.zebra_points)]))
                    cv2.line(display_frame, start, end, (0, 255, 255), 2)
            
            # Draw lines for road edges
            if len(self.road_points) >= 2:
                # Left edge (points 0,2)
                if len(self.road_points) >= 3:
                    cv2.line(display_frame, tuple(map(int, self.road_points[0])), 
                            tuple(map(int, self.road_points[2])), (255, 255, 0), 2)
                # Right edge (points 1,3)
                if len(self.road_points) >= 4:
                    cv2.line(display_frame, tuple(map(int, self.road_points[1])), 
                            tuple(map(int, self.road_points[3])), (255, 255, 0), 2)
            
            # Show current step and instructions
            total_points = len(self.zebra_points) + len(self.road_points)
            if len(self.zebra_points) < 4:
                step_text = f"STEP 1: Zebra crossing corner {len(self.zebra_points)+1}/4"
                color = (0, 255, 0)
            else:
                step_text = f"STEP 2: Road reference point {len(self.road_points)+1}/4"
                color = (255, 0, 0)
            
            instructions = [
                step_text,
                f"Total points: {total_points}/8",
                "r: Reset | s: Save | q: Quit"
            ]
            
            for i, instruction in enumerate(instructions):
                text_color = color if i == 0 else (255, 255, 255)
                cv2.putText(display_frame, instruction, 
                           (20, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            cv2.imshow('Road Alignment Setup', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.zebra_points = []
                self.road_points = []
                self.setup_step = "zebra"
                print("All points reset")
            elif key == ord('s') and len(self.zebra_points) == 4 and len(self.road_points) == 4:
                self.zebra_configured = True
                self.compute_homography_enhanced()
                self.save_zebra_config()
                print("Road alignment configuration saved!")
                break
            elif key == ord('s'):
                print("Please select all 8 points (4 zebra + 4 road reference points)")
        
        cv2.destroyWindow('Road Alignment Setup')
        return self.zebra_configured
    
    def mouse_callback_enhanced(self, event, x, y, flags, param):
        """Handle mouse clicks for zebra crossing and road reference point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.zebra_points) < 4:
                self.zebra_points.append([x, y])
                print(f"Zebra point {len(self.zebra_points)}: ({x}, {y})")
            elif len(self.road_points) < 4:
                self.road_points.append([x, y])
                print(f"Road reference point {len(self.road_points)}: ({x}, {y})")
    
    def toggle_adjustment_mode(self):
        """Toggle real-time adjustment mode on/off."""
        self.adjustment_mode = not self.adjustment_mode
        status = "ENABLED" if self.adjustment_mode else "DISABLED"
        print(f"Real-time homography adjustment: {status}")
        return self.adjustment_mode
    
    def toggle_reference_lines(self):
        """Toggle visibility of reference lines."""
        self.show_reference_lines = not self.show_reference_lines
        status = "VISIBLE" if self.show_reference_lines else "HIDDEN"
        print(f"Reference lines: {status}")
        return self.show_reference_lines
    
    def setup_safety_zones_interactive(self, frame):
        """Setup safety zones for vehicle proximity detection."""
        print("\n=== SAFETY ZONE SETUP ===")
        print("Click 4 points to define the safety zone boundaries:")
        print("1-2: Top boundary line (left to right)")
        print("3-4: Bottom boundary line (left to right)")
        print("Press 's' to save, 'r' to reset, 'q' to quit without saving")
        
        self.safety_zone_top = []
        self.safety_zone_bottom = []
        temp_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                total_points = len(self.safety_zone_top) + len(self.safety_zone_bottom)
                
                if total_points < 2:
                    # Adding top boundary points
                    self.safety_zone_top.append([x, y])
                    print(f"Top boundary point {len(self.safety_zone_top)}: ({x}, {y})")
                elif total_points < 4:
                    # Adding bottom boundary points
                    self.safety_zone_bottom.append([x, y])
                    print(f"Bottom boundary point {len(self.safety_zone_bottom)}: ({x}, {y})")
                
                # Redraw frame with current points
                temp_frame = frame.copy()
                self._draw_safety_zone_setup(temp_frame)
                cv2.imshow('Safety Zone Setup', temp_frame)
        
        cv2.namedWindow('Safety Zone Setup', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Safety Zone Setup', mouse_callback)
        cv2.imshow('Safety Zone Setup', temp_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(self.safety_zone_top) == 2 and len(self.safety_zone_bottom) == 2:
                self.safety_zone_configured = True
                self.save_zebra_config()  # Save with safety zones
                print("Safety zones saved successfully!")
                break
            elif key == ord('r'):
                self.safety_zone_top = []
                self.safety_zone_bottom = []
                temp_frame = frame.copy()
                cv2.imshow('Safety Zone Setup', temp_frame)
                print("Safety zones reset")
            elif key == ord('q'):
                print("Safety zone setup cancelled")
                break
        
        cv2.destroyWindow('Safety Zone Setup')
    
    def _draw_safety_zone_setup(self, frame):
        """Draw safety zone setup visualization."""
        # Draw zebra crossing points if available
        if len(self.zebra_points) == 4:
            zebra_pts = np.array(self.zebra_points, dtype=np.int32)
            cv2.polylines(frame, [zebra_pts], True, (0, 255, 255), 2)  # Yellow zebra crossing
        
        # Draw top boundary line
        if len(self.safety_zone_top) >= 2:
            cv2.line(frame, tuple(self.safety_zone_top[0]), tuple(self.safety_zone_top[1]), (0, 0, 255), 3)  # Red top line
            cv2.putText(frame, "TOP BOUNDARY", tuple(self.safety_zone_top[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw bottom boundary line
        if len(self.safety_zone_bottom) >= 2:
            cv2.line(frame, tuple(self.safety_zone_bottom[0]), tuple(self.safety_zone_bottom[1]), (255, 0, 0), 3)  # Blue bottom line
            cv2.putText(frame, "BOTTOM BOUNDARY", tuple(self.safety_zone_bottom[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw current points
        for i, point in enumerate(self.safety_zone_top):
            cv2.circle(frame, tuple(point), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"T{i+1}", (point[0]+10, point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        for i, point in enumerate(self.safety_zone_bottom):
            cv2.circle(frame, tuple(point), 8, (255, 0, 0), -1)
            cv2.putText(frame, f"B{i+1}", (point[0]+10, point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    def toggle_safety_zones(self):
        """Toggle visibility of safety zone lines."""
        self.show_safety_zones = not self.show_safety_zones
        status = "VISIBLE" if self.show_safety_zones else "HIDDEN"
        print(f"Safety zones: {status}")
        return self.show_safety_zones
    
    def is_vehicle_in_safety_zone(self, x, y):
        """Check if a vehicle position is within the safety zone boundaries OR in the zebra crossing."""
        # First check if vehicle is directly in the zebra crossing
        if self.zebra_configured and len(self.zebra_points) == 4:
            # Get zebra crossing bounds
            zebra_bounds = self.get_zebra_crossing_bounds()
            if zebra_bounds:
                x1, y1, x2, y2 = zebra_bounds
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return True  # Vehicle is in zebra crossing, definitely unsafe
        
        # Then check safety zone boundaries
        if not self.safety_zone_configured or len(self.safety_zone_top) != 2 or len(self.safety_zone_bottom) != 2:
            return False
        
        # PROJECT safety zone coordinates from camera view to top-view coordinates
        if self.homography_matrix is None:
            return False  # Can't project without homography matrix
        
        try:
            # Project safety zone top line points
            top_src_points = np.array(self.safety_zone_top, dtype=np.float32)
            top_projected = cv2.perspectiveTransform(top_src_points.reshape(-1, 1, 2), self.homography_matrix)
            top_line_projected = top_projected.reshape(-1, 2)
            
            # Project safety zone bottom line points
            bottom_src_points = np.array(self.safety_zone_bottom, dtype=np.float32)
            bottom_projected = cv2.perspectiveTransform(bottom_src_points.reshape(-1, 1, 2), self.homography_matrix)
            bottom_line_projected = bottom_projected.reshape(-1, 2)
            
            # Get the projected coordinates
            x1_top, y1_top = top_line_projected[0]
            x2_top, y2_top = top_line_projected[1]
            x1_bottom, y1_bottom = bottom_line_projected[0]
            x2_bottom, y2_bottom = bottom_line_projected[1]
            
        except Exception as e:
            print(f"Error projecting safety zones: {e}")
            return False
        
        # Skip vehicles with clearly invalid projected coordinates
        if x < -100 or x > self.topview_size[1] + 100 or y < -100 or y > self.topview_size[0] + 100:
            return False
        
        # Calculate y-coordinate on top line at vehicle's x position
        if x2_top != x1_top:
            m_top = (y2_top - y1_top) / (x2_top - x1_top)
            y_top = m_top * (x - x1_top) + y1_top
        else:
            y_top = y1_top  # Vertical line
        
        # Calculate y-coordinate on bottom line at vehicle's x position
        if x2_bottom != x1_bottom:
            m_bottom = (y2_bottom - y1_bottom) / (x2_bottom - x1_bottom)
            y_bottom = m_bottom * (x - x1_bottom) + y1_bottom
        else:
            y_bottom = y1_bottom  # Vertical line
        
        # Determine which line is actually the top and which is the bottom based on y-coordinates
        # The line with SMALLER y-coordinate is actually the top (closer to zebra)
        # The line with LARGER y-coordinate is actually the bottom (farther from zebra)
        actual_top_y = min(y_top, y_bottom)
        actual_bottom_y = max(y_top, y_bottom)
        
        # Check if vehicle is within the safety zone boundaries (between actual top and bottom)
        # No buffer - exact boundary detection
        is_in_safety_zone = actual_top_y <= y <= actual_bottom_y
        
        # Debug output to track safety zone boundaries (now in projected coordinates)
        print(f"Vehicle at ({x:.1f},{y:.1f}) - Projected safety zone: {actual_top_y:.1f} <= {y:.1f} <= {actual_bottom_y:.1f} (no buffer) -> {'IN' if is_in_safety_zone else 'OUT'}")
        
        return is_in_safety_zone
    
    def get_safety_zone_bounds(self):
        """Get safety zone boundaries for the safety policy."""
        if not self.safety_zone_configured:
            return None
        
        return {
            'top_line': self.safety_zone_top,
            'bottom_line': self.safety_zone_bottom,
            'configured': True
        }
    
    def find_nearest_point(self, mouse_x, mouse_y):
        """Find the nearest draggable point to mouse position."""
        min_distance = float('inf')
        nearest_point = None
        nearest_index = -1
        nearest_type = ""
        
        # Check zebra crossing points
        for i, point in enumerate(self.zebra_points):
            distance = np.sqrt((point[0] - mouse_x)**2 + (point[1] - mouse_y)**2)
            if distance < min_distance and distance <= self.drag_threshold:
                min_distance = distance
                nearest_point = point
                nearest_index = i
                nearest_type = "zebra"
        
        # Check road reference points
        for i, point in enumerate(self.road_points):
            distance = np.sqrt((point[0] - mouse_x)**2 + (point[1] - mouse_y)**2)
            if distance < min_distance and distance <= self.drag_threshold:
                min_distance = distance
                nearest_point = point
                nearest_index = i
                nearest_type = "road"
        
        # Check safety zone top points
        for i, point in enumerate(self.safety_zone_top):
            distance = np.sqrt((point[0] - mouse_x)**2 + (point[1] - mouse_y)**2)
            if distance < min_distance and distance <= self.drag_threshold:
                min_distance = distance
                nearest_point = point
                nearest_index = i
                nearest_type = "safety_top"
        
        # Check safety zone bottom points
        for i, point in enumerate(self.safety_zone_bottom):
            distance = np.sqrt((point[0] - mouse_x)**2 + (point[1] - mouse_y)**2)
            if distance < min_distance and distance <= self.drag_threshold:
                min_distance = distance
                nearest_point = point
                nearest_index = i
                nearest_type = "safety_bottom"
        
        return nearest_index, nearest_type if nearest_point is not None else (-1, "")
    
    def handle_mouse_event(self, event, x, y, flags, param):
        """Handle mouse events for real-time adjustment during inference."""
        if not self.adjustment_mode:
            return False  # No changes made
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start dragging if near a point
            self.drag_point_index, self.drag_point_type = self.find_nearest_point(x, y)
            if self.drag_point_index >= 0:
                self.dragging = True
                print(f"Started dragging {self.drag_point_type} point {self.drag_point_index + 1}")
                return True
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Update point position during drag
            if self.drag_point_type == "zebra" and 0 <= self.drag_point_index < len(self.zebra_points):
                self.zebra_points[self.drag_point_index] = [x, y]
                self.recompute_homography()
                return True
            elif self.drag_point_type == "road" and 0 <= self.drag_point_index < len(self.road_points):
                self.road_points[self.drag_point_index] = [x, y]
                self.recompute_homography()
                return True
            elif self.drag_point_type == "safety_top" and 0 <= self.drag_point_index < len(self.safety_zone_top):
                self.safety_zone_top[self.drag_point_index] = [x, y]
                return True
            elif self.drag_point_type == "safety_bottom" and 0 <= self.drag_point_index < len(self.safety_zone_bottom):
                self.safety_zone_bottom[self.drag_point_index] = [x, y]
                return True
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            if self.dragging:
                self.dragging = False
                print(f"Stopped dragging {self.drag_point_type} point {self.drag_point_index + 1}")
                # Auto-save configuration after adjustment
                self.save_zebra_config()
                return True
        
        return False  # No changes made
    
    def recompute_homography(self):
        """Recompute homography matrix after point adjustment."""
        if len(self.road_points) == 4:
            self.compute_homography_enhanced()
        elif len(self.zebra_points) == 4:
            self.compute_homography()
    
    def draw_reference_lines(self, frame):
        """Draw reference lines and points on the frame for real-time adjustment."""
        if not self.show_reference_lines:
            return frame
        
        display_frame = frame.copy()
        
        # Draw zebra crossing points and lines (green)
        if len(self.zebra_points) > 0:
            for i, point in enumerate(self.zebra_points):
                # Highlight if being dragged
                color = (0, 255, 255) if (self.dragging and self.drag_point_type == "zebra" and self.drag_point_index == i) else (0, 255, 0)
                thickness = 3 if (self.dragging and self.drag_point_type == "zebra" and self.drag_point_index == i) else 2
                
                cv2.circle(display_frame, tuple(map(int, point)), 8, color, -1)
                cv2.putText(display_frame, f"Z{i+1}", 
                           (int(point[0]+12), int(point[1]-8)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            
            # Draw zebra crossing outline
            if len(self.zebra_points) == 4:
                for i in range(4):
                    start = tuple(map(int, self.zebra_points[i]))
                    end = tuple(map(int, self.zebra_points[(i+1) % 4]))
                    cv2.line(display_frame, start, end, (0, 255, 255), 2)
        
        # Draw road reference points and lines (blue)
        if len(self.road_points) > 0:
            for i, point in enumerate(self.road_points):
                # Highlight if being dragged
                color = (255, 255, 0) if (self.dragging and self.drag_point_type == "road" and self.drag_point_index == i) else (255, 0, 0)
                thickness = 3 if (self.dragging and self.drag_point_type == "road" and self.drag_point_index == i) else 2
                
                cv2.circle(display_frame, tuple(map(int, point)), 8, color, -1)
                cv2.putText(display_frame, f"R{i+1}", 
                           (int(point[0]+12), int(point[1]-8)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            
            # Draw road edge lines
            if len(self.road_points) >= 3:
                # Left edge (points 0,2)
                cv2.line(display_frame, tuple(map(int, self.road_points[0])), 
                        tuple(map(int, self.road_points[2])), (255, 255, 0), 2)
            if len(self.road_points) >= 4:
                # Right edge (points 1,3)
                cv2.line(display_frame, tuple(map(int, self.road_points[1])), 
                        tuple(map(int, self.road_points[3])), (255, 255, 0), 2)
        
        # Add adjustment mode indicator
        mode_text = "ADJUSTMENT MODE: ON" if self.adjustment_mode else "ADJUSTMENT MODE: OFF"
        mode_color = (0, 255, 0) if self.adjustment_mode else (128, 128, 128)
        cv2.putText(display_frame, mode_text, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Add instructions
        if self.adjustment_mode:
            instructions = [
                "Click and drag points to adjust",
                "Press 'a' to toggle adjustment mode",
                "Press 'l' to toggle reference lines",
                "Press 'z' to toggle safety zones"
            ]
        else:
            instructions = [
                "Press 'a' to enable adjustment mode",
                "Press 'l' to toggle reference lines",
                "Press 'z' to toggle safety zones"
            ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, (10, frame.shape[0] - 30 + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw safety zones if configured and visible
        if self.show_safety_zones and self.safety_zone_configured:
            # Draw top boundary line
            if len(self.safety_zone_top) == 2:
                cv2.line(display_frame, tuple(map(int, self.safety_zone_top[0])), 
                        tuple(map(int, self.safety_zone_top[1])), (0, 0, 255), 3)  # Red line
                cv2.putText(display_frame, "SAFETY TOP", tuple(map(int, self.safety_zone_top[0])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw safety zone top points
                for i, point in enumerate(self.safety_zone_top):
                    color = (255, 255, 0) if (self.dragging and self.drag_point_type == "safety_top" and self.drag_point_index == i) else (0, 0, 255)
                    cv2.circle(display_frame, tuple(map(int, point)), 8, color, -1)
                    cv2.putText(display_frame, f"ST{i+1}", (int(point[0])+10, int(point[1])-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw bottom boundary line
            if len(self.safety_zone_bottom) == 2:
                cv2.line(display_frame, tuple(map(int, self.safety_zone_bottom[0])), 
                        tuple(map(int, self.safety_zone_bottom[1])), (255, 0, 0), 3)  # Blue line
                cv2.putText(display_frame, "SAFETY BOTTOM", tuple(map(int, self.safety_zone_bottom[0])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Draw safety zone bottom points
                for i, point in enumerate(self.safety_zone_bottom):
                    color = (255, 255, 0) if (self.dragging and self.drag_point_type == "safety_bottom" and self.drag_point_index == i) else (255, 0, 0)
                    cv2.circle(display_frame, tuple(map(int, point)), 8, color, -1)
                    cv2.putText(display_frame, f"SB{i+1}", (int(point[0])+10, int(point[1])-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_frame
    
    def compute_homography_enhanced(self):
        """Compute homography matrix using only road reference points for better alignment."""
        if len(self.road_points) == 4:
            # Use only road reference points for homography computation
            road_src_points = np.array(self.road_points, dtype=np.float32)
            
            # Map road edges to full canvas corners for maximum accuracy
            self.homography_matrix = cv2.getPerspectiveTransform(road_src_points, self.topview_road_refs)
            print("Enhanced homography matrix computed using 4 road reference points")
            return True
        return False
    
    def compute_homography(self):
        """Compute homography matrix from zebra crossing points (fallback)."""
        if len(self.zebra_points) == 4:
            src_points = np.array(self.zebra_points, dtype=np.float32)
            self.homography_matrix = cv2.getPerspectiveTransform(src_points, self.topview_zebra_rect)
            print("Homography matrix computed successfully")
            return True
        return False
    
    def get_zebra_crossing_bounds(self):
        """Get zebra crossing bounds in top-view coordinates for safety analysis."""
        if not self.zebra_configured or len(self.zebra_points) != 4:
            return None
            
        if self.homography_matrix is not None:
            # Project the actual zebra crossing points to top-view
            zebra_src_points = np.array(self.zebra_points, dtype=np.float32)
            zebra_projected = cv2.perspectiveTransform(zebra_src_points.reshape(-1, 1, 2), self.homography_matrix)
            zebra_rect = zebra_projected.reshape(-1, 2)
            
            # Check if projected zebra crossing is within reasonable bounds
            x_coords = zebra_rect[:, 0]
            y_coords = zebra_rect[:, 1]
            
            # Only return projected bounds if they're within the canvas area (with some margin)
            if (min(x_coords) >= -100 and max(x_coords) <= self.topview_size[1] + 100 and
                min(y_coords) >= -100 and max(y_coords) <= self.topview_size[0] + 100):
                
                # Return bounding box of the projected zebra crossing
                return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        # Fallback to default centered zebra crossing bounds
        zebra_rect = self.topview_zebra_rect
        return (min(zebra_rect[:, 0]), min(zebra_rect[:, 1]), 
                max(zebra_rect[:, 0]), max(zebra_rect[:, 1]))
    
    def project_detections_to_topview(self, detections):
        """Project detections using homography transformation."""
        projected_detections = {
            'positions': [],
            'class_names': detections['class_names'],
            'confidences': detections['confidences']
        }
        
        if self.homography_matrix is None:
            # Fallback to simple scaling if no homography
            for box in detections['boxes']:
                x1, y1, x2, y2 = box
                bottom_center_x = (x1 + x2) / 2
                bottom_center_y = y2
                projected_detections['positions'].append((bottom_center_x, bottom_center_y))
        else:
            # Use homography transformation
            for box in detections['boxes']:
                x1, y1, x2, y2 = box
                bottom_center = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                
                # Apply homography transformation
                projected_point = cv2.perspectiveTransform(bottom_center.reshape(1, 1, 2), self.homography_matrix)
                projected_x, projected_y = projected_point[0, 0]
                
                projected_detections['positions'].append((projected_x, projected_y))
        
        return projected_detections
    
    
    def create_topview_visualization(self, projected_detections, frame_width=1872, frame_height=998):
        """Create top-down view visualization with zebra crossing."""
        # Create blank canvas for top view
        topview_img = np.zeros((self.topview_size[0], self.topview_size[1], 3), dtype=np.uint8)
        
        # Define areas for complete street visualization
        sidewalk_width = 30  # Width of sidewalk areas
        road_margin = 50
        
        # Draw sidewalks first (light gray background for entire area)
        cv2.rectangle(topview_img, (0, 0), (self.topview_size[1], self.topview_size[0]), (120, 120, 120), -1)  # Light gray sidewalks
        
        # Draw left sidewalk area (darker gray)
        cv2.rectangle(topview_img, 
                     (0, 0), 
                     (road_margin + sidewalk_width, self.topview_size[0]),
                     (100, 100, 100), -1)  # Darker gray for left sidewalk
        
        # Draw right sidewalk area (darker gray)
        cv2.rectangle(topview_img, 
                     (self.topview_size[1] - road_margin - sidewalk_width, 0), 
                     (self.topview_size[1], self.topview_size[0]),
                     (100, 100, 100), -1)  # Darker gray for right sidewalk
        
        # Draw main road area
        cv2.rectangle(topview_img, 
                     (road_margin + sidewalk_width, road_margin), 
                     (self.topview_size[1] - road_margin - sidewalk_width, self.topview_size[0] - road_margin),
                     (60, 60, 60), -1)  # Darker gray road
        
        # Add sidewalk labels
        cv2.putText(topview_img, "SIDEWALK", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(topview_img, "SIDEWALK", (self.topview_size[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw zebra crossing if configured - SINGLE zebra crossing only
        zebra_drawn = False
        if self.zebra_configured and len(self.zebra_points) == 4:
            # Try to project zebra crossing points using the homography matrix
            if self.homography_matrix is not None:
                zebra_src_points = np.array(self.zebra_points, dtype=np.float32)
                zebra_projected = cv2.perspectiveTransform(zebra_src_points.reshape(-1, 1, 2), self.homography_matrix)
                zebra_rect = zebra_projected.reshape(-1, 2).astype(int)
                
                # Check if projected zebra crossing is within reasonable bounds
                x_coords = zebra_rect[:, 0]
                y_coords = zebra_rect[:, 1]
                
                # Only draw projected zebra if it's within the canvas area (with some margin)
                if (min(x_coords) >= -100 and max(x_coords) <= self.topview_size[1] + 100 and
                    min(y_coords) >= -100 and max(y_coords) <= self.topview_size[0] + 100):
                    
                    # Draw projected zebra crossing as white stripes
                    cv2.fillPoly(topview_img, [zebra_rect], (200, 200, 200))  # Light gray base
                    
                    # Draw zebra stripes
                    stripe_width = 15
                    stripe_gap = 10
                    x_start, x_end = min(zebra_rect[:, 0]), max(zebra_rect[:, 0])
                    y_start, y_end = min(zebra_rect[:, 1]), max(zebra_rect[:, 1])
                    
                    for x in range(x_start, x_end, stripe_width + stripe_gap):
                        stripe_rect = np.array([
                            [x, y_start],
                            [min(x + stripe_width, x_end), y_start],
                            [min(x + stripe_width, x_end), y_end],
                            [x, y_end]
                        ])
                        cv2.fillPoly(topview_img, [stripe_rect], (255, 255, 255))  # White stripes
                    
                    # Add zebra crossing label
                    label_x = max(0, min(zebra_rect[0, 0], self.topview_size[1] - 150))
                    label_y = max(20, zebra_rect[0, 1] - 10)
                    cv2.putText(topview_img, "ZEBRA CROSSING", 
                               (label_x, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    zebra_drawn = True
            
            # Only use fallback if projected zebra was not drawn
            if not zebra_drawn:
                # Use fallback centered zebra crossing
                zebra_rect = self.topview_zebra_rect.astype(int)
                cv2.fillPoly(topview_img, [zebra_rect], (200, 200, 200))  # Light gray base
                
                # Draw zebra stripes
                stripe_width = 15
                stripe_gap = 10
                x_start, x_end = zebra_rect[0, 0], zebra_rect[1, 0]
                y_start, y_end = zebra_rect[0, 1], zebra_rect[3, 1]
                
                for x in range(x_start, x_end, stripe_width + stripe_gap):
                    stripe_rect = np.array([
                        [x, y_start],
                        [min(x + stripe_width, x_end), y_start],
                        [min(x + stripe_width, x_end), y_end],
                        [x, y_end]
                    ])
                    cv2.fillPoly(topview_img, [stripe_rect], (255, 255, 255))  # White stripes
                
                # Add zebra crossing label
                fallback_reason = "NO HOMOGRAPHY" if self.homography_matrix is None else "OUT OF BOUNDS"
                cv2.putText(topview_img, f"ZEBRA CROSSING ({fallback_reason})", 
                           (zebra_rect[0, 0], zebra_rect[0, 1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                zebra_drawn = True
        
        # Draw center line
        center_x = self.topview_size[1] // 2
        cv2.line(topview_img, 
                (center_x, road_margin), 
                (center_x, self.topview_size[0] - road_margin),
                (255, 255, 255), 2)
        
        # Draw safety zones if configured and visible
        if self.show_safety_zones and self.safety_zone_configured and self.homography_matrix is not None:
            if len(self.safety_zone_top) == 2 and len(self.safety_zone_bottom) == 2:
                # Project safety zone lines to top-view
                try:
                    # Top line
                    top_src_points = np.array(self.safety_zone_top, dtype=np.float32)
                    top_projected = cv2.perspectiveTransform(top_src_points.reshape(-1, 1, 2), self.homography_matrix)
                    top_line = top_projected.reshape(-1, 2).astype(int)
                    
                    # Bottom line
                    bottom_src_points = np.array(self.safety_zone_bottom, dtype=np.float32)
                    bottom_projected = cv2.perspectiveTransform(bottom_src_points.reshape(-1, 1, 2), self.homography_matrix)
                    bottom_line = bottom_projected.reshape(-1, 2).astype(int)
                    
                    # Draw safety zone boundaries
                    cv2.line(topview_img, tuple(top_line[0]), tuple(top_line[1]), (0, 0, 255), 3)  # Red top line
                    cv2.line(topview_img, tuple(bottom_line[0]), tuple(bottom_line[1]), (255, 0, 0), 3)  # Blue bottom line
                    
                    # Add labels
                    cv2.putText(topview_img, "SAFETY TOP", tuple(top_line[0]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(topview_img, "SAFETY BOTTOM", tuple(bottom_line[0]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                except:
                    # If projection fails, skip safety zone drawing
                    pass
        
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
            
            if self.homography_matrix is not None:
                # Use direct projected coordinates
                canvas_x, canvas_y = int(x), int(y)
            else:
                # Fallback scaling including sidewalk areas
                sidewalk_width = 30
                total_margin = road_margin + sidewalk_width
                canvas_x = int((x / frame_width) * (self.topview_size[1] - 2 * total_margin)) + total_margin
                canvas_y = int((y / frame_height) * (self.topview_size[0] - 2 * road_margin)) + road_margin
            
            # Ensure coordinates are within full canvas bounds (including sidewalks)
            canvas_x = max(0, min(self.topview_size[1] - 1, canvas_x))
            canvas_y = max(0, min(self.topview_size[0] - 1, canvas_y))
            
            # Draw object as circle
            cv2.circle(topview_img, (canvas_x, canvas_y), 12, color, -1)
            
            # Add class label
            cv2.putText(topview_img, class_name[:3], 
                       (canvas_x + 15, canvas_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add coordinate information near each vehicle
            coord_text = f"({int(x)},{int(y)})"
            cv2.putText(topview_img, coord_text, 
                       (canvas_x + 15, canvas_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # For vehicles, also show if they're in safety zone
            if class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                if self.safety_zone_configured:
                    is_in_zone = self.is_vehicle_in_safety_zone(x, y)
                    zone_status = "IN" if is_in_zone else "OUT"
                    zone_color = (0, 255, 0) if is_in_zone else (0, 0, 255)
                    cv2.putText(topview_img, zone_status, 
                               (canvas_x + 15, canvas_y + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_color, 1)
        
        # Add safety zone coordinate information at the top
        if self.safety_zone_configured and len(self.safety_zone_top) == 2 and len(self.safety_zone_bottom) == 2:
            # Display safety zone y-coordinates at different x positions
            test_x_positions = [200, 300, 400, 500]
            y_offset = 160
            
            cv2.putText(topview_img, "Safety Zone Y-coords at different X:", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            for i, test_x in enumerate(test_x_positions):
                # Calculate y-coordinates of safety lines at this x position
                x1_top, y1_top = self.safety_zone_top[0]
                x2_top, y2_top = self.safety_zone_top[1]
                x1_bottom, y1_bottom = self.safety_zone_bottom[0]
                x2_bottom, y2_bottom = self.safety_zone_bottom[1]
                
                # Calculate y-coordinate on top line
                if x2_top != x1_top:
                    m_top = (y2_top - y1_top) / (x2_top - x1_top)
                    y_top_at_x = m_top * (test_x - x1_top) + y1_top
                else:
                    y_top_at_x = y1_top
                
                # Calculate y-coordinate on bottom line
                if x2_bottom != x1_bottom:
                    m_bottom = (y2_bottom - y1_bottom) / (x2_bottom - x1_bottom)
                    y_bottom_at_x = m_bottom * (test_x - x1_bottom) + y1_bottom
                else:
                    y_bottom_at_x = y1_bottom
                
                zone_text = f"X{test_x}: {int(min(y_top_at_x, y_bottom_at_x))}-{int(max(y_top_at_x, y_bottom_at_x))}"
                cv2.putText(topview_img, zone_text, 
                           (20 + i * 150, y_offset + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add title and info
        cv2.putText(topview_img, "Top View (Homography)", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        total_objects = len(projected_detections['positions'])
        cv2.putText(topview_img, f"Objects: {total_objects}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add configuration status
        status = "Configured" if self.zebra_configured else "Not Configured"
        cv2.putText(topview_img, f"Zebra: {status}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0) if self.zebra_configured else (0, 0, 255), 2)
        
        return topview_img
