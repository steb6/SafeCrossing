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
        
        # Step 4: Analyze safety and get recommendations
        safety_result = self.safety_policy.analyze_safety(
            projected_detections, 
            self.current_traffic_light_state,
            zebra_bounds
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
    
    def _log_status_change(self, new_status, safety_result, frame_number):
        """Log detailed status changes with reasoning including early warning vehicles."""
        risk_level = safety_result['risk_level']
        recommendations = safety_result.get('recommendations', [])
        
        # Get detailed information including new early warning category
        pedestrians_in_crossing = safety_result.get('pedestrians_in_crossing', 0)
        vehicles_early_warning = safety_result.get('vehicles_early_warning', 0)
        vehicles_approaching = safety_result.get('vehicles_approaching', 0)
        vehicles_close_approach = safety_result.get('vehicles_close_approach', 0)
        vehicles_high_speed_close = safety_result.get('vehicles_high_speed_close', 0)
        total_vehicles = safety_result.get('vehicles_near_crossing', 0)
        
        # Status change announcement
        status_icon = {"SAFE": "✅", "NOT_SAFE": "⚠️", "DANGER": "🚨"}.get(new_status.value, "❓")
        print(f"\n{status_icon} FRAME {frame_number}: STATUS CHANGED TO {new_status.value}")
        print(f"   Risk Level: {risk_level:.2f}")
        
        # Detailed reasoning based on status
        if new_status.value == "DANGER":
            print("   🚨 DANGER REASONS:")
            if vehicles_high_speed_close > 0:
                print(f"   • {vehicles_high_speed_close} vehicle(s) at HIGH SPEED very close to zebra crossing")
            if vehicles_close_approach > 0 and pedestrians_in_crossing > 0:
                print(f"   • {vehicles_close_approach} vehicle(s) close approach + {pedestrians_in_crossing} pedestrian(s) crossing")
            if risk_level >= 0.8:
                print("   • Risk level indicates immediate collision threat")
                
        elif new_status.value == "NOT_SAFE":
            print("   ⚠️ NOT_SAFE REASONS:")
            if vehicles_early_warning > 0:
                print(f"   • 🟡 {vehicles_early_warning} vehicle(s) in early warning zone (300px)")
            if vehicles_approaching > 0:
                print(f"   • 🟠 {vehicles_approaching} vehicle(s) approaching zebra crossing (200px)")
            if vehicles_close_approach > 0:
                print(f"   • 🔴 {vehicles_close_approach} vehicle(s) in close approach zone (100px)")
            if pedestrians_in_crossing > 0 and total_vehicles > 0:
                print(f"   • {pedestrians_in_crossing} pedestrian(s) crossing with {total_vehicles} vehicle(s) nearby")
                
        elif new_status.value == "SAFE":
            print("   ✅ SAFE REASONS:")
            if total_vehicles == 0:
                print("   • No vehicles detected near zebra crossing")
            elif pedestrians_in_crossing == 0:
                print("   • No pedestrians currently crossing")
            else:
                print("   • Risk factors below safety thresholds")
        
        # Show main recommendation
        if recommendations:
            print(f"   📋 ACTION: {recommendations[0]}")
        
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
                        
                        # Update frame counter overlay to show current frame number
                        actual_height, actual_width = combined_frame.shape[:2]
                        cv2.putText(combined_frame, f"Frame: {self.frame_count} (SKIPPED)", 
                                   (actual_width-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                    else:
                        # Fallback if no previous frame processed yet
                        combined_frame = frame
                    
                    processing_time = 0.001  # Minimal processing time
                    pipeline_result = last_pipeline_result  # Use last pipeline result
                
                # Get the actual dimensions of the combined frame
                actual_height, actual_width = combined_frame.shape[:2]
                
                # Add FPS info to the combined frame
                fps_value = 1/processing_time if processing_time > 0 else 0
                cv2.putText(combined_frame, f"FPS: {fps_value:.1f}", 
                           (actual_width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined_frame, f"Frame: {self.frame_count}", 
                           (actual_width-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Print safety alerts (only for processed frames)
                if pipeline_result is not None and frame_counter % frame_skip == 0:
                    safety_status = pipeline_result['safety_result']['status']
                    
                    # Check for status changes and provide detailed reasoning
                    if last_safety_status is None or safety_status != last_safety_status:
                        self._log_status_change(safety_status, pipeline_result['safety_result'], self.frame_count)
                        last_safety_status = safety_status
            
            # Display the frame
            cv2.namedWindow('Smart Traffic Light System', cv2.WINDOW_NORMAL)
            cv2.imshow('Smart Traffic Light System', combined_frame)
            
            # Handle key presses with proper frame timing
            if paused:
                # When paused, wait indefinitely until key press
                key = cv2.waitKey(0) & 0xFF
            else:
                # When playing, wait for frame delay to maintain proper FPS
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save detection cache
        print("Saving detection cache...")
        self.detector.save_cache()
        
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
