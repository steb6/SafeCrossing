"""
Safety Policy Module for Smart Traffic Light System
Comprehensive safety analysis based on detected objects and zebra crossing.
"""

from enum import Enum
import numpy as np


class SafetyStatus(Enum):
    """Safety status levels."""
    SAFE = "SAFE"
    NOT_SAFE = "NOT_SAFE"
    DANGER = "DANGER"


class SafetyPolicy:
    """Comprehensive safety analysis for traffic control."""
    
    def __init__(self):
        """Initialize safety policy with configurable parameters."""
        print("Safety policy initialized")
        
        # Safety thresholds
        self.pedestrian_danger_zone = 50  # pixels from zebra crossing
        self.vehicle_speed_threshold = 0.8  # relative speed threshold
        self.max_safe_pedestrians = 3  # max pedestrians in crossing
        self.vehicle_stop_distance = 100  # minimum stopping distance
        
        # Risk weights for different scenarios
        self.risk_weights = {
            'pedestrian_in_crossing': 0.8,
            'vehicle_near_crossing': 0.6,
            'high_traffic_density': 0.4,
            'mixed_traffic': 0.3
        }
        
        # Traffic light timing
        self.min_green_time = 30  # minimum green light duration (seconds)
        self.max_red_time = 60   # maximum red light duration (seconds)
        
    def analyze_safety(self, projected_detections, traffic_light_state=None, zebra_crossing_bounds=None):
        """
        Comprehensive safety analysis based on object positions and traffic state.
        
        Args:
            projected_detections (dict): Detected objects with projected positions
            traffic_light_state (str): Current light state (GREEN, YELLOW, RED)
            zebra_crossing_bounds (tuple): Zebra crossing area bounds (x1, y1, x2, y2)
            
        Returns:
            dict: Comprehensive safety analysis results
        """
        class_names = projected_detections.get('class_names', [])
        positions = projected_detections.get('positions', [])
        confidences = projected_detections.get('confidences', [])
        
        # Count different object types
        object_counts = {}
        pedestrians_in_crossing = 0
        vehicles_near_crossing = 0
        high_risk_objects = []
        
        for i, class_name in enumerate(class_names):
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            if len(positions) > i:
                x, y = positions[i]
                confidence = confidences[i] if i < len(confidences) else 0.0
                
                # Analyze pedestrian safety
                if class_name == 'person':
                    if self._is_in_zebra_crossing(x, y, zebra_crossing_bounds):
                        pedestrians_in_crossing += 1
                        high_risk_objects.append({
                            'type': 'pedestrian_in_crossing',
                            'position': (x, y),
                            'confidence': confidence
                        })
                
                # Analyze vehicle safety
                elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
                    if self._is_near_zebra_crossing(x, y, zebra_crossing_bounds):
                        vehicles_near_crossing += 1
                        high_risk_objects.append({
                            'type': 'vehicle_near_crossing',
                            'position': (x, y),
                            'confidence': confidence
                        })
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(
            pedestrians_in_crossing, vehicles_near_crossing, 
            object_counts, traffic_light_state
        )
        
        # Determine safety status
        safety_status = self._determine_safety_status(risk_level, high_risk_objects)
        
        # Generate safety recommendations
        recommendations = self._generate_safety_recommendations(
            safety_status, pedestrians_in_crossing, vehicles_near_crossing,
            traffic_light_state
        )
        
        return {
            'status': safety_status,
            'risk_level': risk_level,
            'object_counts': object_counts,
            'total_objects': len(class_names),
            'pedestrians_in_crossing': pedestrians_in_crossing,
            'vehicles_near_crossing': vehicles_near_crossing,
            'high_risk_objects': high_risk_objects,
            'recommendations': recommendations
        }
    
    def _is_in_zebra_crossing(self, x, y, zebra_bounds):
        """Check if position is within zebra crossing area."""
        if zebra_bounds is None:
            # Default zebra crossing area (centered in top view: 800x600 canvas)
            # Center: (400, 300), Size: 200x60
            return 300 <= x <= 500 and 270 <= y <= 330
        
        x1, y1, x2, y2 = zebra_bounds
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _is_near_zebra_crossing(self, x, y, zebra_bounds):
        """Check if position is near zebra crossing (danger zone)."""
        if zebra_bounds is None:
            # Default danger zone around centered zebra crossing
            margin = self.pedestrian_danger_zone
            return (300 - margin) <= x <= (500 + margin) and (270 - margin) <= y <= (330 + margin)
        
        x1, y1, x2, y2 = zebra_bounds
        margin = self.pedestrian_danger_zone
        return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)
    
    def _calculate_risk_level(self, pedestrians_in_crossing, vehicles_near_crossing, 
                             object_counts, traffic_light_state):
        """Calculate overall risk level (0.0 to 1.0)."""
        risk = 0.0
        
        # Pedestrian risk
        if pedestrians_in_crossing > 0:
            risk += self.risk_weights['pedestrian_in_crossing']
            if pedestrians_in_crossing > self.max_safe_pedestrians:
                risk += 0.3  # Additional risk for overcrowding
        
        # Vehicle risk
        if vehicles_near_crossing > 0:
            risk += self.risk_weights['vehicle_near_crossing']
            if traffic_light_state == "GREEN" and pedestrians_in_crossing > 0:
                risk += 0.4  # High risk: vehicles moving with pedestrians crossing
        
        # Traffic density risk
        total_objects = sum(object_counts.values())
        if total_objects > 10:
            risk += self.risk_weights['high_traffic_density']
        
        # Mixed traffic risk
        has_pedestrians = object_counts.get('person', 0) > 0
        has_vehicles = any(object_counts.get(v, 0) > 0 for v in ['car', 'truck', 'bus', 'motorcycle'])
        if has_pedestrians and has_vehicles:
            risk += self.risk_weights['mixed_traffic']
        
        # Traffic light state modifiers
        if traffic_light_state == "RED" and vehicles_near_crossing > 0:
            risk += 0.2  # Risk of vehicles running red light
        elif traffic_light_state == "YELLOW":
            risk += 0.3  # Uncertain behavior during yellow
        
        return min(risk, 1.0)  # Cap at 1.0
    
    def _determine_safety_status(self, risk_level, high_risk_objects):
        """Determine safety status based on risk level."""
        if risk_level >= 0.7:
            return SafetyStatus.DANGER
        elif risk_level >= 0.4:
            return SafetyStatus.NOT_SAFE
        else:
            return SafetyStatus.SAFE
    
    def _generate_safety_recommendations(self, safety_status, pedestrians_in_crossing, 
                                       vehicles_near_crossing, traffic_light_state):
        """Generate safety recommendations based on current situation."""
        recommendations = []
        
        if safety_status == SafetyStatus.DANGER:
            recommendations.append("IMMEDIATE ACTION REQUIRED")
            if pedestrians_in_crossing > 0 and vehicles_near_crossing > 0:
                recommendations.append("Clear pedestrians from crossing")
                recommendations.append("Stop vehicle traffic immediately")
        
        elif safety_status == SafetyStatus.NOT_SAFE:
            if pedestrians_in_crossing > self.max_safe_pedestrians:
                recommendations.append("Too many pedestrians in crossing")
            if vehicles_near_crossing > 0 and traffic_light_state == "GREEN":
                recommendations.append("Monitor vehicle behavior closely")
        
        if len(recommendations) == 0:
            recommendations.append("Normal operation")
        
        return recommendations
    
    def get_traffic_light_recommendation(self, safety_result, current_phase):
        """
        Generate traffic light control recommendations based on safety analysis.
        
        Args:
            safety_result (dict): Result from analyze_safety()
            current_phase (str): Current traffic light phase
            
        Returns:
            dict: Traffic light recommendation
        """
        safety_status = safety_result['status']
        pedestrians_in_crossing = safety_result['pedestrians_in_crossing']
        vehicles_near_crossing = safety_result['vehicles_near_crossing']
        risk_level = safety_result['risk_level']
        
        # Emergency override for dangerous situations
        if safety_status == SafetyStatus.DANGER:
            if pedestrians_in_crossing > 0:
                return {
                    'action': 'EMERGENCY_STOP',
                    'recommendation': 'Stop all traffic - pedestrians in danger',
                    'priority': 'CRITICAL'
                }
        
        # High risk situations
        if safety_status == SafetyStatus.NOT_SAFE:
            if current_phase == "GREEN" and pedestrians_in_crossing > 0:
                return {
                    'action': 'EXTEND_RED',
                    'recommendation': 'Extend red light - allow pedestrians to clear',
                    'priority': 'HIGH'
                }
            elif current_phase == "RED" and vehicles_near_crossing > 2:
                return {
                    'action': 'MONITOR',
                    'recommendation': 'Monitor for red light violations',
                    'priority': 'MEDIUM'
                }
        
        # Normal operation recommendations
        if pedestrians_in_crossing > 0:
            return {
                'action': 'CONTINUE',
                'recommendation': f'Allow {pedestrians_in_crossing} pedestrian(s) to cross',
                'priority': 'LOW'
            }
        
        # Default recommendation
        return {
            'action': 'CONTINUE',
            'recommendation': 'Normal traffic light operation',
            'priority': 'LOW'
        }
