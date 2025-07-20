from enum import Enum
import numpy as np


class SafetyStatus(Enum):
    """Safety status levels."""
    SAFE = "SAFE"
    NOT_SAFE = "NOT_SAFE"
    DANGER = "DANGER"


class SafetyPolicy:
    """Simplified but working safety policy for smart traffic systems."""
    
    def _init_(self):
        """Initialize safety policy."""
        print("Safety policy initialized (working version)")
    
    def analyze_safety(self, projected_detections, traffic_light_state=None, zebra_crossing_bounds=None, homography_projector=None):
        """
        Perform simple safety analysis based on detected objects and positions.
        
        Args:
            projected_detections (dict): Detected objects with projected positions
            traffic_light_state (str): Current light state (GREEN, YELLOW, RED)
            zebra_crossing_bounds (tuple): Zebra crossing area bounds (x1, y1, x2, y2)
            homography_projector: HomographyProjector instance for safety zone checking
            
        Returns:
            dict: Safety analysis results
        """
        class_names = projected_detections.get('class_names', [])
        positions = projected_detections.get('positions', [])
        
        object_counts = {}
        for class_name in class_names:
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        vehicle_positions = []
        pedestrian_positions = []
        
        # Track detected vehicles to avoid double-counting people on vehicles
        vehicle_classes_detected = set()
        for class_name in class_names:
            if class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                vehicle_classes_detected.add(class_name)
        
        for i, class_name in enumerate(class_names):
            if i < len(positions):
                x, y = positions[i]
                if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_positions.append((x, y))
                elif class_name == 'bicycle':
                    # Bicycles are also vehicles for safety purposes
                    vehicle_positions.append((x, y))
                elif class_name == 'person':
                    # Only count as pedestrian if no vehicles are detected nearby
                    # This helps avoid counting motorcycle/bicycle riders as pedestrians
                    is_vehicle_rider = False
                    
                    # Check if there are any vehicles very close to this person (likely a rider)
                    for j, other_class in enumerate(class_names):
                        if j != i and other_class in ['motorcycle', 'bicycle'] and j < len(positions):
                            other_x, other_y = positions[j]
                            # If person is within 50 pixels of a motorcycle/bicycle, likely a rider
                            distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                            if distance < 50:  # Close proximity threshold
                                is_vehicle_rider = True
                                break
                    
                    # Only add as pedestrian if not likely a vehicle rider
                    if not is_vehicle_rider:
                        pedestrian_positions.append((x, y))

        safety_status, extra_info = self._analyze_traffic_safety(
            vehicle_positions, 
            pedestrian_positions, 
            zebra_crossing_bounds,
            homography_projector
        )
        
        return {
            'status': safety_status,
            'risk_level': 1.0 if safety_status == SafetyStatus.DANGER else (0.5 if safety_status == SafetyStatus.NOT_SAFE else 0.0),
            'object_counts': object_counts,
            'total_objects': len(class_names),
            'pedestrians_in_crossing': extra_info['pedestrians_in_crossing'],
            'vehicles_near_crossing': extra_info['vehicles_near_crossing'],
            'vehicles_in_crossing': extra_info['vehicles_in_crossing'],
            'recommendations': [f"🔴 {safety_status.value}: Use caution" if safety_status != SafetyStatus.SAFE else "✅ Safe to proceed"]
        }
    
    def _analyze_traffic_safety(self, vehicle_positions, pedestrian_positions, zebra_crossing_bounds, homography_projector=None):
        """
        Analyze the safety of the scene based on positions and crossing area.
        Uses safety zones if configured, otherwise falls back to threshold-based detection.
        """
        x1, y1, x2, y2 = zebra_crossing_bounds if zebra_crossing_bounds else (0, 0, 0, 0)
        
        def is_inside_crossing(x, y):
            return x1 <= x <= x2 and y1 <= y <= y2
        
        def is_near_crossing_fallback(x, y, threshold=30):
            # Fallback method when safety zones are not configured
            cx = max(min(x, x2), x1)
            cy = max(min(y, y2), y1)
            return np.linalg.norm(np.array([x, y]) - np.array([cx, cy])) < threshold

        pedestrians_in_crossing = sum(1 for (x, y) in pedestrian_positions if is_inside_crossing(x, y))
        vehicles_in_crossing = sum(1 for (x, y) in vehicle_positions if is_inside_crossing(x, y))
        
        # Filter out invalid coordinates (negative values or extreme outliers)
        valid_vehicle_positions = [(x, y) for (x, y) in vehicle_positions 
                                 if x >= 0 and y >= 0 and x < 2000 and y < 2000]
        
        # Use safety zones if configured, otherwise fall back to threshold-based detection
        if homography_projector and homography_projector.safety_zone_configured:
            vehicles_near_crossing = sum(1 for (x, y) in valid_vehicle_positions 
                                       if homography_projector.is_vehicle_in_safety_zone(x, y))
        else:
            vehicles_near_crossing = sum(1 for (x, y) in valid_vehicle_positions if is_near_crossing_fallback(x, y))

        # DANGER: Vehicle directly on zebra crossing OR pedestrians crossing with vehicles nearby
        if vehicles_in_crossing > 0:
            return SafetyStatus.DANGER, {
                'pedestrians_in_crossing': pedestrians_in_crossing,
                'vehicles_near_crossing': vehicles_near_crossing,
                'vehicles_in_crossing': vehicles_in_crossing
            }
        elif pedestrians_in_crossing > 0 and vehicles_near_crossing > 0:
            return SafetyStatus.DANGER, {
                'pedestrians_in_crossing': pedestrians_in_crossing,
                'vehicles_near_crossing': vehicles_near_crossing,
                'vehicles_in_crossing': vehicles_in_crossing
            }
        elif vehicles_near_crossing > 0:
            # NOT_SAFE when vehicles are near crossing, regardless of pedestrian presence
            return SafetyStatus.NOT_SAFE, {
                'pedestrians_in_crossing': pedestrians_in_crossing,
                'vehicles_near_crossing': vehicles_near_crossing,
                'vehicles_in_crossing': vehicles_in_crossing
            }
        else:
            return SafetyStatus.SAFE, {
                'pedestrians_in_crossing': pedestrians_in_crossing,
                'vehicles_near_crossing': vehicles_near_crossing,
                'vehicles_in_crossing': vehicles_in_crossing
            }

    def get_traffic_light_recommendation(self, safety_result, current_phase):
        """
        Simple traffic light recommendation based on safety status.
        """
        status = safety_result.get("status")
        if status == SafetyStatus.DANGER:
            return "RED"
        elif status == SafetyStatus.NOT_SAFE:
            return "YELLOW"
        else:
            return "GREEN"