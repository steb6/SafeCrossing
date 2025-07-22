"""
Smart Traffic Light System - Main Pipeline
Integrates detection, homography projection, and safety policy modules.
"""

import cv2
import json
import os
import time
import numpy as np
from detection import TrafficDetector
from homography import HomographyProjector
from safety_policy import SafetyPolicy, SafetyStatus


class SmartTrafficLightSystem:
    """Main pipeline for smart traffic light control."""
    
    def __init__(self, model_name="yolo11x.pt", confidence_threshold=0.3, cache_dir="detection_cache"):
        """Initialize the smart traffic light system."""
        print("Initializing Smart Traffic Light System...")
        
        # Initialize modules
        self.detector = TrafficDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            cache_dir=cache_dir
        )
        self.homography = HomographyProjector()
        self.safety_policy = SafetyPolicy()
        
        # System state
        self.current_traffic_light_state = "GREEN"  # GREEN, YELLOW, RED
        self.frame_count = 0
        self.zebra_setup_done = False
        
        # Display layout properties for coordinate mapping
        self.original_frame_width = 0
        self.combined_frame_width = 0
        
        # Manual labeling system for ground truth
        self.ground_truth_file = "ground_truth_labels.json"
        self.ground_truth_labels = {}  # frame_number -> label
        self.current_label = "SAFE"  # Default label
        self.labeling_mode = False
        self.load_ground_truth_labels()
        
        print("System initialized successfully!")
    
    def load_ground_truth_labels(self):
        """Load existing ground truth labels from file."""
        if os.path.exists(self.ground_truth_file):
            try:
                with open(self.ground_truth_file, 'r') as f:
                    self.ground_truth_labels = json.load(f)
                print(f"Loaded {len(self.ground_truth_labels)} ground truth labels from {self.ground_truth_file}")
                self.labeling_mode = False  # Use existing labels for evaluation
            except Exception as e:
                print(f"Error loading ground truth labels: {e}")
                self.ground_truth_labels = {}
                self.labeling_mode = True  # Start labeling mode
        else:
            print(f"No ground truth file found. Starting labeling mode.")
            self.ground_truth_labels = {}
            self.labeling_mode = True
    
    def save_ground_truth_labels(self):
        """Save ground truth labels to file."""
        try:
            with open(self.ground_truth_file, 'w') as f:
                json.dump(self.ground_truth_labels, f, indent=2)
            print(f"Saved {len(self.ground_truth_labels)} ground truth labels to {self.ground_truth_file}")
        except Exception as e:
            print(f"Error saving ground truth labels: {e}")
    
    def get_ground_truth_label(self, frame_number):
        """Get ground truth label for a specific frame."""
        return self.ground_truth_labels.get(str(frame_number), None)
    
    def set_ground_truth_label(self, frame_number, label):
        """Set ground truth label for a specific frame and fill in the range automatically."""
        # Find the last state change before this frame
        previous_frames = [int(f) for f in self.ground_truth_labels.keys() if int(f) < frame_number]
        
        if previous_frames:
            # Get the most recent labeled frame
            last_labeled_frame = max(previous_frames)
            last_label = self.ground_truth_labels[str(last_labeled_frame)]
            
            # Fill in all frames between the last labeled frame and current frame with the previous label
            for f in range(last_labeled_frame + 1, frame_number):
                self.ground_truth_labels[str(f)] = last_label
                
            print(f"Auto-filled frames {last_labeled_frame + 1} to {frame_number - 1} with {last_label}")
        else:
            # No previous labels, fill from frame 181 (after skipped frames) to current frame with default SAFE
            start_frame = 181  # After the 180 skipped frames
            for f in range(start_frame, frame_number):
                self.ground_truth_labels[str(f)] = "SAFE"
                
            if frame_number > start_frame:
                print(f"Auto-filled frames {start_frame} to {frame_number - 1} with SAFE (default)")
        
        # Set the current frame with the new label
        self.ground_truth_labels[str(frame_number)] = label
        self.current_label = label
        print(f"Frame {frame_number}: State changed to {label}")
        print(f"All subsequent frames will be labeled as {label} until next change")
    
    def get_current_ground_truth_state(self, frame_number):
        """Get the current ground truth state for a frame based on range logic."""
        # If we have a specific label for this frame, use it
        if str(frame_number) in self.ground_truth_labels:
            return self.ground_truth_labels[str(frame_number)]
        
        # Otherwise, find the most recent state change before this frame
        relevant_frames = [int(f) for f in self.ground_truth_labels.keys() if int(f) <= frame_number]
        
        if relevant_frames:
            # Get the most recent frame with a label
            most_recent_frame = max(relevant_frames)
            return self.ground_truth_labels[str(most_recent_frame)]
        else:
            # No previous labels, default to SAFE (for frames before first user input)
            return "SAFE"
    
    def finalize_ground_truth_labels(self, final_frame_number):
        """Fill in any remaining unlabeled frames at the end of the video."""
        if not self.labeling_mode:
            return
            
        # Find the last labeled frame
        if self.ground_truth_labels:
            labeled_frames = [int(f) for f in self.ground_truth_labels.keys()]
            last_labeled_frame = max(labeled_frames)
            last_label = self.ground_truth_labels[str(last_labeled_frame)]
            
            # Fill in all remaining frames with the last label
            frames_filled = 0
            for f in range(last_labeled_frame + 1, final_frame_number + 1):
                self.ground_truth_labels[str(f)] = last_label
                frames_filled += 1
                
            if frames_filled > 0:
                print(f"Auto-filled final {frames_filled} frames ({last_labeled_frame + 1} to {final_frame_number}) with {last_label}")
                self.save_ground_truth_labels()
        else:
            # No labels at all, fill everything with SAFE
            start_frame = 181  # After the 180 skipped frames
            for f in range(start_frame, final_frame_number + 1):
                self.ground_truth_labels[str(f)] = "SAFE"
            print(f"No manual labels provided. Auto-filled all frames ({start_frame} to {final_frame_number}) with SAFE")
            self.save_ground_truth_labels()
    
    def toggle_labeling_mode(self):
        """Toggle between labeling mode and evaluation mode."""
        self.labeling_mode = not self.labeling_mode
        mode = "LABELING" if self.labeling_mode else "EVALUATION"
        print(f"Switched to {mode} mode")
        return self.labeling_mode
    
    def handle_mouse_for_homography(self, event, x, y, flags, param):
        """Handle mouse events with coordinate mapping for the combined display."""
        # Check if click is within the original frame area (left side)
        if x < self.original_frame_width:
            # Pass the coordinates to homography handler (they're already in original frame coordinates)
            return self.homography.handle_mouse_event(event, x, y, flags, param)
        else:
            # Click was in the top-view area (right side), ignore it for homography adjustment
            return False
    
    def setup_zebra_crossing(self, video_path):
        """Setup zebra crossing if not already configured."""
        if not self.homography.zebra_configured:
            print("Zebra crossing not configured. Starting interactive setup...")
            
            # Get first frame for setup
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self.homography.setup_zebra_crossing_interactive(frame)
                self.zebra_setup_done = True
                
                # After zebra crossing setup, optionally setup safety zones
                if self.zebra_setup_done and not self.homography.safety_zone_configured:
                    print("\nWould you like to setup safety zones for vehicle detection? (y/n)")
                    choice = input().lower().strip()
                    if choice in ['y', 'yes']:
                        self.homography.setup_safety_zones_interactive(frame)
            else:
                print("Could not read video for zebra crossing setup")
        else:
            print("Zebra crossing already configured")
            self.zebra_setup_done = True
            
            # Check if safety zones need setup
            if not self.homography.safety_zone_configured:
                print("Safety zones not configured. Would you like to set them up? (y/n)")
                choice = input().lower().strip()
                if choice in ['y', 'yes']:
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        self.homography.setup_safety_zones_interactive(frame)
    
    def process_frame(self, frame, frame_number=None):
        """Process a single video frame through the complete pipeline."""
        
        # Step 1: Detect traffic objects with caching
        if frame_number is not None:
            detections = self.detector.detect_with_cache(frame, frame_number)
            # If cache-only mode and no cache available, return None to signal stop
            if detections is None:
                return None
        else:
            # Fallback to regular detection if frame number not provided
            detections = self.detector.detect(frame)
        
        # Step 2: Project to top view
        projected_detections = self.homography.project_detections_to_topview(detections)
        
        # Step 3: Get actual zebra crossing bounds for safety analysis
        zebra_bounds = self.homography.get_zebra_crossing_bounds()
        
        # Step 4: Analyze safety and get recommendations (pass homography projector for safety zones)
        safety_result = self.safety_policy.analyze_safety(
            projected_detections, 
            self.current_traffic_light_state,
            zebra_bounds,
            self.homography  # Pass homography projector for safety zone checking
        )
        
        # Step 5: Get traffic light control recommendation
        traffic_recommendation = self.safety_policy.get_traffic_light_recommendation(
            safety_result, 
            self.current_traffic_light_state
        )
        
        return {
            'detections': detections,
            'projected_detections': projected_detections,
            'safety_result': safety_result,
            'traffic_recommendation': traffic_recommendation
        }
    
    def visualize_results(self, frame, pipeline_result):
        """Create visualization combining all pipeline results."""
        
        # Start with detection visualization
        annotated_frame = self.detector.visualize_detections(
            frame, pipeline_result['detections']
        )
        
        # Add homography reference lines for real-time adjustment
        annotated_frame = self.homography.draw_reference_lines(annotated_frame)
        
        # Add safety status overlay
        safety_status = pipeline_result['safety_result']['status']
        risk_level = pipeline_result['safety_result']['risk_level']
        pedestrians_in_crossing = pipeline_result['safety_result'].get('pedestrians_in_crossing', 0)
        vehicles_near_crossing = pipeline_result['safety_result'].get('vehicles_near_crossing', 0)
        vehicles_in_crossing = pipeline_result['safety_result'].get('vehicles_in_crossing', 0)
        
        # Choose color based on safety status
        if safety_status == SafetyStatus.SAFE:
            status_color = (0, 255, 0)  # Green
        elif safety_status == SafetyStatus.NOT_SAFE:
            status_color = (0, 255, 255)  # Yellow
        else:  # DANGER
            status_color = (0, 0, 255)  # Red
        
        # Map safety status to traffic light state
        if safety_status == SafetyStatus.SAFE:
            traffic_light_state = "GREEN"
        elif safety_status == SafetyStatus.NOT_SAFE:
            traffic_light_state = "YELLOW"
        else:  # DANGER
            traffic_light_state = "RED"
        
        # Update current state for internal logic
        self.current_traffic_light_state = traffic_light_state
        
        # Extra large safety status panel with massive text
        cv2.rectangle(annotated_frame, (10, 10), (950, 430), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Safety: {safety_status.value}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.5, status_color, 8)
        cv2.putText(annotated_frame, f"Risk Level: {risk_level:.2f}", 
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        
        # Add pedestrian and vehicle counts with massive text
        cv2.putText(annotated_frame, f"Pedestrians in crossing: {pedestrians_in_crossing}", 
                   (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
        cv2.putText(annotated_frame, f"Vehicles near crossing: {vehicles_near_crossing}", 
                   (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
        cv2.putText(annotated_frame, f"Vehicles in crossing: {vehicles_in_crossing}", 
                   (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0) if vehicles_in_crossing > 0 else (255, 255, 255), 5)
        
        # Add object count with larger text size
        object_counts = {}
        for class_name in pipeline_result['detections']['class_names']:
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        y_offset = 340  # Moved up since we removed the traffic light state text
        for class_name, count in object_counts.items():
            cv2.putText(annotated_frame, f"{class_name}: {count}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
            y_offset += 55  # Increased spacing between lines
        
        # Add performance evaluation overlay in top right corner of left panel
        frame_height, frame_width = annotated_frame.shape[:2]
        eval_x = frame_width - 700  # Position from right edge - made even wider
        eval_y = 100  # Start from top - bigger margin
        
        # Get ground truth and predicted labels
        ground_truth_label = self.get_current_ground_truth_state(self.frame_count)
        predicted_label = "SAFE" if safety_status == SafetyStatus.SAFE else "NOT_SAFE"
        
        # Calculate cumulative accuracy
        if hasattr(self, 'total_frames_evaluated'):
            self.total_frames_evaluated += 1
            if ground_truth_label == predicted_label:
                self.correct_predictions += 1
        else:
            self.total_frames_evaluated = 1
            self.correct_predictions = 1 if ground_truth_label == predicted_label else 0
        
        accuracy = (self.correct_predictions / self.total_frames_evaluated) * 100 if self.total_frames_evaluated > 0 else 0
        
        # Draw evaluation panel background - much bigger
        cv2.rectangle(annotated_frame, (eval_x - 50, eval_y - 50), (frame_width - 10, eval_y + 280), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (eval_x - 50, eval_y - 50), (frame_width - 10, eval_y + 280), (100, 100, 100), 5)
        
        # Ground truth label - massive font with bold stroke
        gt_color = (0, 255, 0) if ground_truth_label == "SAFE" else (0, 0, 255)
        cv2.putText(annotated_frame, f"GT: {ground_truth_label}", (eval_x, eval_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, gt_color, 8)
        eval_y += 80  # Much bigger spacing
        
        # Predicted label - massive font with bold stroke
        pred_color = (0, 255, 0) if predicted_label == "SAFE" else (0, 0, 255)
        cv2.putText(annotated_frame, f"Pred: {predicted_label}", (eval_x, eval_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, pred_color, 8)
        eval_y += 80  # Much bigger spacing
        
        # Accuracy - massive font with bold stroke, white color
        cv2.putText(annotated_frame, f"Acc: {accuracy:.1f}%", (eval_x, eval_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 8)
        
        # Draw traffic light semaphore in center of frame - much bigger and positioned lower
        frame_height, frame_width = annotated_frame.shape[:2]
        center_x = frame_width // 2
        semaphore_y = 300  # Moved down further from 200 to 300
        
        # Add "PREDICTED STATUS" label above the traffic light
        label_y = semaphore_y - 60  # Position above the traffic light
        # Draw black background rectangle for better text readability
        text_size = cv2.getTextSize("PREDICTED STATUS", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
        text_x = center_x - 200
        cv2.rectangle(annotated_frame, 
                     (text_x - 10, label_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, label_y + 10), 
                     (0, 0, 0), -1)  # Black background
        cv2.putText(annotated_frame, "PREDICTED STATUS", 
                   (text_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Traffic light background (black rectangle) - made much bigger
        light_width = 160  # Increased from 120 to 160
        light_height = 450  # Increased from 330 to 450
        cv2.rectangle(annotated_frame, 
                     (center_x - light_width//2, semaphore_y), 
                     (center_x + light_width//2, semaphore_y + light_height), 
                     (30, 30, 30), -1)  # Dark gray background
        
        # Traffic light border
        cv2.rectangle(annotated_frame, 
                     (center_x - light_width//2, semaphore_y), 
                     (center_x + light_width//2, semaphore_y + light_height), 
                     (200, 200, 200), 5)  # Light gray border - even thicker
        
        # Light positions - much bigger spacing and radius
        light_radius = 50  # Increased from 35 to 50
        red_pos = (center_x, semaphore_y + 80)    # Adjusted for much bigger size
        yellow_pos = (center_x, semaphore_y + 225) # Adjusted for much bigger size
        green_pos = (center_x, semaphore_y + 370)  # Adjusted for much bigger size
        
        # Draw all lights (dim when not active)
        # Red light
        red_color = (0, 0, 255) if traffic_light_state == "RED" else (50, 0, 0)
        cv2.circle(annotated_frame, red_pos, light_radius, red_color, -1)
        cv2.circle(annotated_frame, red_pos, light_radius, (100, 100, 100), 4)
        
        # Yellow light
        yellow_color = (0, 255, 255) if traffic_light_state == "YELLOW" else (50, 50, 0)
        cv2.circle(annotated_frame, yellow_pos, light_radius, yellow_color, -1)
        cv2.circle(annotated_frame, yellow_pos, light_radius, (100, 100, 100), 4)
        
        # Green light
        green_color = (0, 255, 0) if traffic_light_state == "GREEN" else (0, 50, 0)
        cv2.circle(annotated_frame, green_pos, light_radius, green_color, -1)
        cv2.circle(annotated_frame, green_pos, light_radius, (100, 100, 100), 4)
        
        # Create top-view visualization
        topview_img = self.homography.create_topview_visualization(
            pipeline_result['projected_detections'], 
            frame.shape[1], frame.shape[0]  # width, height
        )
        
        # Create side-by-side layout without stretching
        frame_height, frame_width = annotated_frame.shape[:2]
        topview_height, topview_width = topview_img.shape[:2]
        
        # Calculate scaling to match heights while preserving aspect ratios
        scale_factor = frame_height / topview_height
        new_topview_width = int(topview_width * scale_factor)
        topview_resized = cv2.resize(topview_img, (new_topview_width, frame_height))
        
        # Create combined frame with both views
        total_width = frame_width + new_topview_width
        combined_frame = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
        
        # Store frame dimensions for coordinate mapping
        self.original_frame_width = frame_width
        self.combined_frame_width = total_width
        
        # Place original frame on the left (no stretching)
        combined_frame[:, :frame_width] = annotated_frame
        
        # Place top-view on the right
        combined_frame[:, frame_width:] = topview_resized
        
        # Add ground truth labeling information if in labeling mode
        if self.labeling_mode:
            # Add labeling instructions overlay
            overlay_y = 50
            cv2.putText(combined_frame, "LABELING MODE", (10, overlay_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            overlay_y += 35
            cv2.putText(combined_frame, f"Frame: {self.frame_count}", (10, overlay_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            overlay_y += 30
            
            # Current ground truth state (automatically determined)
            current_gt_state = self.get_current_ground_truth_state(self.frame_count)
            state_color = (0, 255, 0) if current_gt_state == "SAFE" else (0, 0, 255)
            cv2.putText(combined_frame, f"Current Label: {current_gt_state}", (10, overlay_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            overlay_y += 30
            
            # Show if this frame has an explicit label (state change point)
            explicit_label = self.get_ground_truth_label(self.frame_count)
            if explicit_label:
                cv2.putText(combined_frame, f"[Manual Label Point]", (10, overlay_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                overlay_y += 25
            
            # Show labeling statistics
            total_labeled = len(self.ground_truth_labels)
            cv2.putText(combined_frame, f"Total Labels: {total_labeled}", (10, overlay_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            overlay_y += 25
            
            # Instructions
            cv2.putText(combined_frame, "Auto-Range Labeling:", (10, overlay_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            overlay_y += 25
            cv2.putText(combined_frame, "  + = Label as SAFE from here", (10, overlay_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            overlay_y += 20
            cv2.putText(combined_frame, "  - = Label as NOT_SAFE from here", (10, overlay_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            overlay_y += 20
            cv2.putText(combined_frame, "  t = Toggle Mode", (10, overlay_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return combined_frame
    
    def _log_status_change(self, new_status, safety_result, frame_number):
        """Log concise status changes with key reasons only."""
        # Get essential information
        pedestrians_in_crossing = safety_result.get('pedestrians_in_crossing', 0)
        vehicles_in_crossing = safety_result.get('vehicles_in_crossing', 0)
        total_vehicles = safety_result.get('vehicles_near_crossing', 0)
        
        # Status change announcement with reason
        status_icon = {"SAFE": "✅", "NOT_SAFE": "⚠️", "DANGER": "🚨"}.get(new_status.value, "❓")
        
        # Determine and show the primary reason for the status
        if new_status.value == "DANGER":
            if vehicles_in_crossing > 0:
                reason = f"🚗 {vehicles_in_crossing} vehicle(s) on zebra crossing"
            elif pedestrians_in_crossing > 0 and total_vehicles > 0:
                reason = f"👥 {pedestrians_in_crossing} pedestrian(s) crossing with {total_vehicles} vehicle(s) nearby"
            else:
                reason = "High collision risk detected"
            print(f"{status_icon} FRAME {frame_number}: DANGER - {reason}")
                
        elif new_status.value == "NOT_SAFE":
            if total_vehicles > 0:
                reason = f"� {total_vehicles} vehicle(s) near zebra crossing"
            else:
                reason = "Potential safety concerns"
            print(f"{status_icon} FRAME {frame_number}: NOT_SAFE - {reason}")
                
        elif new_status.value == "SAFE":
            if total_vehicles == 0 and pedestrians_in_crossing == 0:
                reason = "No vehicles or pedestrians detected"
            elif total_vehicles == 0:
                reason = "No vehicles near crossing"
            elif pedestrians_in_crossing == 0:
                reason = "No active pedestrian crossings"
            else:
                reason = "All safety conditions met"
            print(f"{status_icon} FRAME {frame_number}: SAFE - {reason}")
        
        print()  # Empty line for readability
    
    def run_video_processing(self, video_path):
        """Run the complete pipeline on a video file."""
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
            return
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        # Open video stream
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps} FPS")
        
        # Initialize cache for this video
        print("Initializing detection cache for video...")
        self.detector.initialize_cache_for_video(video_path)
        
        # Calculate frame delay for proper playback speed
        target_fps = fps
        frame_delay_ms = int(1000 / target_fps)  # Delay in milliseconds
        
        # Option to reduce processing load for quieter operation
        frame_skip = 1  # Process every frame (set to 2 for half FPS, 3 for third FPS, etc.)
        if self.detector.cache_only_mode:
            frame_skip = 3  # Skip 2 out of 3 frames in cache-only mode for ultra-quiet operation
            frame_delay_ms = int(frame_delay_ms * 1.5)  # Also slow down playback slightly
            print(f"🤫 ULTRA-QUIET MODE: Processing every {frame_skip} frames for minimal CPU load")
        
        print(f"Target FPS: {target_fps}, Frame delay: {frame_delay_ms}ms")
        
        # Skip the first 180 frames (useless frames)
        frames_to_skip = 180
        print(f"Skipping first {frames_to_skip} frames...")
        for i in range(frames_to_skip):
            ret, _ = cap.read()
            if not ret:
                print("Video ended before skipping all frames")
                break
            self.frame_count += 1
        
        print(f"Starting processing from frame {self.frame_count + 1}")
        
        # Processing loop
        paused = False
        frame_counter = 0
        last_combined_frame = None  # Store last processed visualization to prevent flickering
        last_pipeline_result = None
        last_safety_status = None  # Track status changes
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video or failed to read frame")
                    break
                
                self.frame_count += 1
                frame_counter += 1
                
                # Skip frames for reduced processing load
                if frame_counter % frame_skip == 0:
                    # Process frame through pipeline
                    start_time = time.time()
                    pipeline_result = self.process_frame(frame, self.frame_count)
                    
                    # Check if processing should stop (cache-only mode with no more cache)
                    if pipeline_result is None:
                        print("🏁 Reached end of cached detections. Stopping to keep quiet!")
                        break
                        
                    processing_time = time.time() - start_time
                    
                    # Create visualization and store it
                    combined_frame = self.visualize_results(frame, pipeline_result)
                    last_combined_frame = combined_frame.copy()  # Store for skipped frames
                    last_pipeline_result = pipeline_result
                else:
                    # For skipped frames, use the last processed visualization to prevent flickering
                    if last_combined_frame is not None:
                        combined_frame = last_combined_frame.copy()
                        
                        # Update frame counter overlay to show current frame number with huge text
                        actual_height, actual_width = combined_frame.shape[:2]
                        cv2.putText(combined_frame, f"Frame: {self.frame_count} (SKIPPED)", 
                                   (actual_width-500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 4)
                    else:
                        # Fallback if no previous frame processed yet
                        combined_frame = frame
                    
                    processing_time = 0.001  # Minimal processing time
                    pipeline_result = last_pipeline_result  # Use last pipeline result
                
                # Get the actual dimensions of the combined frame
                actual_height, actual_width = combined_frame.shape[:2]
                
                # Add frame counter to the combined frame
                cv2.putText(combined_frame, f"Frame: {self.frame_count}", 
                           (actual_width-400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                
                # Print safety alerts (only for processed frames)
                if pipeline_result is not None and frame_counter % frame_skip == 0:
                    safety_status = pipeline_result['safety_result']['status']
                    
                    # Check for status changes and provide detailed reasoning
                    if last_safety_status is None or safety_status != last_safety_status:
                        # self._log_status_change(safety_status, pipeline_result['safety_result'], self.frame_count)
                        last_safety_status = safety_status
            
            # Display the frame
            cv2.namedWindow('Smart Traffic Light System', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Smart Traffic Light System', self.handle_mouse_for_homography)
            cv2.imshow('Smart Traffic Light System', combined_frame)
            
            # Handle key presses with proper frame timing
            if paused:
                # When paused, wait indefinitely until key press
                key = cv2.waitKey(0) & 0xFF
            else:
                # When playing, wait for frame delay to maintain proper FPS
                key = cv2.waitKey(1) & 0xFF
                
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('a'):
                # Toggle adjustment mode
                self.homography.toggle_adjustment_mode()
            elif key == ord('l'):
                # Toggle reference lines visibility
                self.homography.toggle_reference_lines()
            elif key == ord('s') and self.homography.adjustment_mode:
                # Manual save configuration
                self.homography.save_zebra_config()
                print("Configuration saved manually")
            elif key == ord('z'):
                # Toggle safety zones
                self.homography.toggle_safety_zones()
            elif key == ord('-') or key == ord('_'):
                # Set current frame as NOT_SAFE
                if self.labeling_mode:
                    self.set_ground_truth_label(self.frame_count, "NOT_SAFE")
                    self.current_label = "NOT_SAFE"
                    self.save_ground_truth_labels()
            elif key == ord('+') or key == ord('='):
                # Set current frame as SAFE
                if self.labeling_mode:
                    self.set_ground_truth_label(self.frame_count, "SAFE")
                    self.current_label = "SAFE"
                    self.save_ground_truth_labels()
            elif key == ord('t'):
                # Toggle labeling mode
                self.toggle_labeling_mode()
            elif paused and key == 83:  # Right arrow key (when paused)
                # Move forward one frame
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if current_frame < total_frames - 1:
                    ret, frame = cap.read()
                    if ret:
                        self.frame_count = current_frame + 1
                        print(f"Frame: {self.frame_count}/{total_frames}")
                        # Process this frame to show updated visualization
                        pipeline_result = self.process_frame(frame, self.frame_count)
                        if pipeline_result:
                            combined_frame = self.visualize_results(frame, pipeline_result)
                            cv2.imshow('Smart Traffic Light System', combined_frame)
            elif paused and key == 81:  # Left arrow key (when paused)
                # Move backward one frame
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 2)  # Go back 2 to compensate for the read
                    ret, frame = cap.read()
                    if ret:
                        self.frame_count = current_frame - 1
                        print(f"Frame: {self.frame_count}")
                        # Process this frame to show updated visualization
                        pipeline_result = self.process_frame(frame, self.frame_count)
                        if pipeline_result:
                            combined_frame = self.visualize_results(frame, pipeline_result)
                            cv2.imshow('Smart Traffic Light System', combined_frame)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Finalize ground truth labels if in labeling mode
        if self.labeling_mode:
            self.finalize_ground_truth_labels(self.frame_count)
        
        # Save detection cache
        print("Saving detection cache...")
        self.detector.save_cache()
        
        print(f"Video processing completed! Processed {self.frame_count} frames.")


def main():
    """Main entry point."""
    # Configuration
    video_path = "Kigali.mp4"
    model_name = "yolo11x.pt"  # Use the most accurate model
    confidence_threshold = 0.3
    
    print("Smart Traffic Light System")
    print("=" * 50)
    print("ACVSS25 Hackathon Project - SafeCrossing")
    print("=" * 50)
    print(f"Input video: {video_path}")
    print(f"Model: {model_name}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 50)
    
    # Initialize system
    system = SmartTrafficLightSystem(
        model_name=model_name,
        confidence_threshold=confidence_threshold
    )
    
    # Setup zebra crossing (interactive if not configured)
    print("\nChecking zebra crossing configuration...")
    system.setup_zebra_crossing(video_path)
    
    if not system.zebra_setup_done:
        print("Zebra crossing setup was not completed. Continuing with basic projection.")
    else:
        print("Zebra crossing configured successfully!")
    
    # Run video processing
    print("\nStarting video processing...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  'a' - Toggle adjustment mode (drag points to recalibrate)")
    print("  'l' - Toggle reference lines visibility")
    print("  's' - Save configuration (when in adjustment mode)")
    print("  'z' - Toggle safety zones visibility")
    print("  'Left/Right Arrow' - Navigate frames when paused")
    print("=" * 50)
    
    system.run_video_processing(video_path)


if __name__ == "__main__":
    main()
