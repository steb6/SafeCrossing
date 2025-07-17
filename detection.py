"""
Object Detection Module for Smart Traffic Light System
Handles YOLO-based detection of traffic-relevant objects.
"""

import cv2
from ultralytics import YOLO
import torch


class TrafficDetector:
    """Detects vehicles and road users using YOLO model."""
    
    def __init__(self, model_name="yolo11x.pt", device="auto", confidence_threshold=0.3):
        """
        Initialize the traffic detector.
        
        Args:
            model_name (str): YOLO model to use
            device (str): Device for inference ("auto", "cpu", "cuda", or device number)
            confidence_threshold (float): Minimum confidence for detections
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Setup device
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.device != "cpu":
            self.model.to('cuda')
        
        # COCO class IDs for traffic-relevant objects
        self.traffic_classes = {
            0: 'person',        # Pedestrians
            1: 'bicycle',       # Bicycles
            2: 'car',          # Cars
            3: 'motorcycle',    # Motorcycles  
            4: 'airplane',      # Airplanes (rarely on roads)
            5: 'bus',          # Buses
            6: 'train',        # Trains
            7: 'truck',        # Trucks
        }
        
        print(f"Traffic detector initialized with {model_name} on {self.device}")
    
    def detect(self, frame):
        """
        Detect traffic-relevant objects in a frame.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            dict: Detection results with filtered bounding boxes and metadata
        """
        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        detections = {
            'boxes': [],
            'classes': [],
            'confidences': [],
            'class_names': []
        }
        
        # Filter results for traffic-relevant objects only
        if result.boxes is not None:
            class_ids = result.boxes.cls.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
            confidences = result.boxes.conf.cpu().numpy()
            
            # Filter for traffic classes
            for i, cls_id in enumerate(class_ids):
                if int(cls_id) in self.traffic_classes:
                    detections['boxes'].append(boxes[i])
                    detections['classes'].append(int(cls_id))
                    detections['confidences'].append(confidences[i])
                    detections['class_names'].append(self.traffic_classes[int(cls_id)])
        
        return detections
    
    def visualize_detections(self, frame, detections):
        """
        Draw bounding boxes on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            detections (dict): Detection results from detect()
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Color map for different classes
        colors = {
            'person': (0, 255, 0),      # Green for pedestrians
            'bicycle': (255, 0, 0),     # Blue for bicycles  
            'car': (0, 0, 255),         # Red for cars
            'motorcycle': (255, 0, 255), # Magenta for motorcycles
            'bus': (0, 255, 255),       # Yellow for buses
            'truck': (128, 0, 128),     # Purple for trucks
            'train': (255, 255, 0),     # Cyan for trains
            'airplane': (128, 128, 128)  # Gray for airplanes
        }
        
        for i, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            class_name = detections['class_names'][i]
            confidence = detections['confidences'][i]
            
            # Get color for this class
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
