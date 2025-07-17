from ultralytics import YOLO
import cv2
import os

# Load the best pretrained YOLO11 model (yolo11x.pt is the most accurate)
model = YOLO("yolo11x.pt")  # Using YOLOv11x for best accuracy

# Use GPU for faster inference
model.to('cuda')  # Move model to GPU

# Path to the video file
video_path = "AbbeyRoad.mp4"

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found!")
    exit(1)

print(f"Processing video: {video_path}")
print("Press 'q' to quit the video playback")

# Open video stream
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width}x{height} @ {fps} FPS")

# Process video frame by frame in real-time
frame_count = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or failed to read frame")
        break
    
    frame_count += 1
    
    # Run YOLO inference on current frame using GPU
    results = model.predict(
        source=frame,
        conf=0.3,  # Confidence threshold
        device=0,  # Use first GPU
        verbose=False  # Reduce console output
    )
    
    # Filter results for traffic-relevant objects only
    # COCO class IDs for vehicles and road users
    traffic_classes = {
        0: 'person',        # Pedestrians
        1: 'bicycle',       # Bicycles
        2: 'car',          # Cars
        3: 'motorcycle',    # Motorcycles  
        4: 'airplane',      # Airplanes (rarely on roads but kept for completeness)
        5: 'bus',          # Buses
        6: 'train',        # Trains
        7: 'truck',        # Trucks
        # Note: We exclude other classes like bags, ties, animals, etc.
    }
    
    # Get the original result
    result = results[0]
    
    # Filter detections to keep only traffic-relevant classes
    if result.boxes is not None:
        # Get class IDs from detections
        class_ids = result.boxes.cls.cpu().numpy()
        
        # Create mask for traffic-relevant classes
        traffic_mask = [int(cls_id) in traffic_classes for cls_id in class_ids]
        
        # Filter boxes, confidences, and class IDs
        if any(traffic_mask):
            result.boxes.data = result.boxes.data[traffic_mask]
        else:
            # No traffic objects detected, create empty boxes
            result.boxes = None
    
    # Draw predictions on the frame (now filtered)
    annotated_frame = result.plot()
    
    # Ensure the annotated frame maintains original dimensions
    if annotated_frame.shape[:2] != (height, width):
        annotated_frame = cv2.resize(annotated_frame, (width, height))
    
    # Display the frame with predictions
    cv2.namedWindow('YOLO Real-time Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO Real-time Detection', width, height)
    cv2.imshow('YOLO Real-time Detection', annotated_frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Optional: Print progress every 30 frames
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Video processing completed! Processed {frame_count} frames.")
