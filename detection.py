"""
Object Detection Module for Smart Traffic Light System
Handles YOLO-based detection of traffic-relevant objects with caching support.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json
import os
import hashlib


class TrafficDetector:
    """Detects vehicles and road users using YOLO model."""
    
    def __init__(self, model_name="yolo11x.pt", device="auto", confidence_threshold=0.3, cache_dir="detection_cache"):
        """
        Initialize the traffic detector.
        
        Args:
            model_name (str): YOLO model to use
            device (str): Device for inference ("auto", "cpu", "cuda", or device number)
            confidence_threshold (float): Minimum confidence for detections
            cache_dir (str): Directory to store detection cache files
        """
        # Store model parameters but don't load the model yet
        self.model_name = model_name
        self.model = None  # Will be loaded lazily when needed
        self.confidence_threshold = confidence_threshold
        self.cache_dir = cache_dir
        self.detection_cache = {}
        self.video_hash = None
        self.model_loaded = False
        self.cache_only_mode = False  # If True, never load model and stop when cache runs out
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Setup device
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
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
        
        print(f"Traffic detector initialized (model will be loaded only if needed)")
    
    def _load_model_if_needed(self):
        """Load the YOLO model only when actually needed for detection."""
        if self.cache_only_mode:
            print("🤫 Cache-only mode: Model loading blocked to keep quiet!")
            return False
            
        if not self.model_loaded:
            print(f"Loading YOLO model {self.model_name}...")
            self.model = YOLO(self.model_name)
            
            if self.device != "cpu":
                self.model.to('cuda')
            
            self.model_loaded = True
            print(f"Model loaded on {self.device}")
        
        return True
    
    def initialize_cache_for_video(self, video_path):
        """
        Initialize detection cache for a specific video file.
        
        Args:
            video_path (str): Path to the video file
        """
        # Generate a unique hash for the video file
        self.video_hash = self._get_video_hash(video_path)
        cache_file = os.path.join(self.cache_dir, f"{self.video_hash}_detections.json")
        
        # Load existing cache if available
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    cached_detections = cache_data.get('detections', {})
                    
                    # Convert lists back to NumPy arrays for compatibility
                    self.detection_cache = {}
                    for frame_key, detections in cached_detections.items():
                        converted_detections = {
                            'boxes': [np.array(box, dtype=np.float32) for box in detections['boxes']],
                            'classes': detections['classes'],
                            'confidences': [np.float32(conf) for conf in detections['confidences']],
                            'class_names': detections['class_names']
                        }
                        self.detection_cache[frame_key] = converted_detections
                    
                    print(f"🤫 SILENT MODE: Using cached detections ({len(self.detection_cache)} frames)")
                    print(f"📁 Cache file: {cache_file}")
                    print(f"🔇 Model loading SKIPPED to keep your computer quiet during class!")
                    
                    # Mark that we should NOT load the model
                    self.cache_only_mode = True
                        
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.detection_cache = {}
                self.cache_only_mode = False
        else:
            self.detection_cache = {}
            self.cache_only_mode = False
            print(f"No cache found for video {video_path}. Model will be loaded for detection.")
    
    def save_cache(self):
        """Save the current detection cache to disk."""
        if self.video_hash and self.detection_cache:
            cache_file = os.path.join(self.cache_dir, f"{self.video_hash}_detections.json")
            
            # Convert NumPy arrays to lists for JSON serialization
            serializable_cache = {}
            for frame_key, detections in self.detection_cache.items():
                serializable_detections = {
                    'boxes': [box.tolist() if hasattr(box, 'tolist') else list(box) for box in detections['boxes']],
                    'classes': detections['classes'],  # Already int
                    'confidences': [float(conf) if hasattr(conf, 'item') else float(conf) for conf in detections['confidences']],
                    'class_names': detections['class_names']  # Already strings
                }
                serializable_cache[frame_key] = serializable_detections
            
            cache_data = {
                'video_hash': self.video_hash,
                'model_name': self.model_name,  # Use stored model name instead of model.model
                'confidence_threshold': self.confidence_threshold,
                'total_frames': len(self.detection_cache),
                'detections': serializable_cache
            }
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                print(f"Saved detection cache with {len(self.detection_cache)} frames to {cache_file}")
            except Exception as e:
                print(f"Error saving cache: {e}")
    
    def _get_video_frame_count(self, video_path):
        """
        Get the total number of frames in a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            int: Total number of frames, or 0 if error
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count
        except Exception as e:
            print(f"Error getting frame count: {e}")
            return 0
    
    def _get_video_hash(self, video_path):
        """
        Generate a unique hash for a video file based on its path and size.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Unique hash for the video
        """
        try:
            # Include file path, size, and modification time for uniqueness
            stat = os.stat(video_path)
            content = f"{video_path}_{stat.st_size}_{stat.st_mtime}_{self.confidence_threshold}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            print(f"Error generating video hash: {e}")
            return hashlib.md5(video_path.encode()).hexdigest()
    
    def detect_with_cache(self, frame, frame_number):
        """
        Detect objects with caching support.
        
        Args:
            frame (np.ndarray): Input video frame
            frame_number (int): Frame number for caching
            
        Returns:
            dict: Detection results, or None if cache-only mode and no cache available
        """
        # Check if detection is cached
        frame_key = str(frame_number)
        if frame_key in self.detection_cache:
            return self.detection_cache[frame_key]
        
        # If we're in cache-only mode and no cache available, return None to signal stop
        if self.cache_only_mode:
            print(f"🛑 Cache-only mode: Frame {frame_number} not in cache. Stopping to keep quiet!")
            return None
        
        # Run detection and cache the result (only if not in cache-only mode)
        detections = self.detect(frame)
        self.detection_cache[frame_key] = detections
        
        return detections
    
    def detect(self, frame):
        """
        Detect traffic-relevant objects in a frame.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            dict: Detection results with filtered bounding boxes and metadata
        """
        # Load model only when needed
        if not self._load_model_if_needed():
            # Model loading was blocked (cache-only mode)
            return {
                'boxes': [],
                'classes': [],
                'confidences': [],
                'class_names': []
            }
        
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
