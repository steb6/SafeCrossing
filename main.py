"""
Smart Traffic Light System - Main Pipeline
Integrates detection, homography projection, and safety policy modules.
"""

import cv2
import os
import time
import numpy as np
from detection import TrafficDetector
from homography import HomographyProjector
from safety_policy import SafetyPolicy, SafetyStatus


class SmartTrafficLightSystem:
    """Main pipeline for smart traffic light control."""
    
    def __init__(self, model_name="yolo11x.pt", confidence_threshold=0.3):
        """Initialize the smart traffic light system."""
        print("Initializing Smart Traffic Light System...")
        
        # Initialize modules
        self.detector = TrafficDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )
        self.homography = HomographyProjector()
        self.safety_policy = SafetyPolicy()
        
        # System state
        self.current_traffic_light_state = "GREEN"  # GREEN, YELLOW, RED
        self.frame_count = 0
        self.zebra_setup_done = False
        
        print("System initialized successfully!")
    
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
            else:
                print("Could not read video for zebra crossing setup")
        else:
            print("Zebra crossing already configured")
            self.zebra_setup_done = True
    
    def process_frame(self, frame):
        """Process a single video frame through the complete pipeline."""
        
        # Step 1: Detect traffic objects
        detections = self.detector.detect(frame)
        
        # Step 2: Project to top view
        projected_detections = self.homography.project_detections_to_topview(detections)
        
        # Step 3: Analyze safety and get recommendations
        safety_result = self.safety_policy.analyze_safety(
            projected_detections, 
            self.current_traffic_light_state
        )
        
        # Step 4: Get traffic light control recommendation
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
        
        # Add safety status overlay
        safety_status = pipeline_result['safety_result']['status']
        risk_level = pipeline_result['safety_result']['risk_level']
        pedestrians_in_crossing = pipeline_result['safety_result'].get('pedestrians_in_crossing', 0)
        vehicles_near_crossing = pipeline_result['safety_result'].get('vehicles_near_crossing', 0)
        
        # Choose color based on safety status
        if safety_status == SafetyStatus.SAFE:
            status_color = (0, 255, 0)  # Green
        elif safety_status == SafetyStatus.NOT_SAFE:
            status_color = (0, 255, 255)  # Yellow
        else:  # DANGER
            status_color = (0, 0, 255)  # Red
        
        # Larger safety status panel
        cv2.rectangle(annotated_frame, (10, 10), (500, 160), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Safety: {safety_status.value}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(annotated_frame, f"Risk Level: {risk_level:.2f}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add pedestrian and vehicle counts
        cv2.putText(annotated_frame, f"Pedestrians in crossing: {pedestrians_in_crossing}", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Vehicles near crossing: {vehicles_near_crossing}", 
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add traffic light state
        light_color = (0, 255, 0) if self.current_traffic_light_state == "GREEN" else \
                     (0, 255, 255) if self.current_traffic_light_state == "YELLOW" else \
                     (0, 0, 255)
        cv2.putText(annotated_frame, f"Light: {self.current_traffic_light_state}", 
                   (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)
        
        # Add object count with bigger, more visible text
        object_counts = {}
        for class_name in pipeline_result['detections']['class_names']:
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        y_offset = 180
        for class_name, count in object_counts.items():
            cv2.putText(annotated_frame, f"{class_name}: {count}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 35  # More spacing between lines
        
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
        
        # Place original frame on the left (no stretching)
        combined_frame[:, :frame_width] = annotated_frame
        
        # Place top-view on the right
        combined_frame[:, frame_width:] = topview_resized
        
        return combined_frame
    
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
        
        # Processing loop
        paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video or failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Process frame through pipeline
                start_time = time.time()
                pipeline_result = self.process_frame(frame)
                processing_time = time.time() - start_time
                
                # Create visualization
                combined_frame = self.visualize_results(frame, pipeline_result)
                
                # Get the actual dimensions of the combined frame
                actual_height, actual_width = combined_frame.shape[:2]
                
                # Add FPS info to the combined frame
                cv2.putText(combined_frame, f"FPS: {1/processing_time:.1f}", 
                           (actual_width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined_frame, f"Frame: {self.frame_count}", 
                           (actual_width-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Print safety alerts
                safety_status = pipeline_result['safety_result']['status']
                if safety_status != SafetyStatus.SAFE:
                    print(f"Frame {self.frame_count}: {safety_status.value} - "
                          f"Risk: {pipeline_result['safety_result']['risk_level']:.2f}")
                
                # Progress reporting
                if self.frame_count % 30 == 0:
                    print(f"Processed {self.frame_count} frames... "
                          f"Current status: {safety_status.value}")
            
            # Display the frame
            cv2.namedWindow('Smart Traffic Light System', cv2.WINDOW_NORMAL)
            cv2.imshow('Smart Traffic Light System', combined_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"Video processing completed! Processed {self.frame_count} frames.")


def main():
    """Main entry point."""
    # Configuration
    video_path = "AbbeyRoad.mp4"
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
    print("=" * 50)
    
    system.run_video_processing(video_path)


if __name__ == "__main__":
    main()
