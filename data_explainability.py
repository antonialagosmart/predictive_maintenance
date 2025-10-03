"""
Data Explainability Module
Validates sensor inputs in real-time and provides transparent data quality alerts
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

class SensorValidator:
    """Real-time sensor data validation and explainability"""
    
    # Define normal operating ranges and critical thresholds
    SENSOR_RANGES = {
        'TP2': {'min': 6.0, 'max': 10.0, 'critical_low': 2.0, 'critical_high': 9.5, 'unit': 'bar'},
        'TP3': {'min': 7.0, 'max': 11.0, 'critical_low': 6.0, 'critical_high': 10.0, 'unit': 'bar'},
        'H1': {'min': 0.1, 'max': 2.0, 'critical_low': 0.05, 'critical_high': 1.8, 'unit': 'bar'},
        'DV_pressure': {'min': 0.0, 'max': 2.5, 'critical_low': -0.1, 'critical_high': 2.0, 'unit': 'bar'},
        'Reservoirs': {'min': 7.0, 'max': 11.0, 'critical_low': 6.5, 'critical_high': 10.5, 'unit': 'bar'},
        'Motor_current': {'min': 0.0, 'max': 9.0, 'critical_low': 0.5, 'critical_high': 8.5, 'unit': 'A'},
        'Oil_temperature': {'min': 40.0, 'max': 65.0, 'critical_low': 25.0, 'critical_high': 70.0, 'unit': '°C'}
    }
    
    # Correlation rules for cross-sensor validation
    CORRELATION_RULES = [
        {
            'sensors': ['TP3', 'Reservoirs'],
            'rule': 'should_match',
            'tolerance': 0.5,
            'description': 'TP3 and Reservoirs pressure should closely match'
        },
        {
            'sensors': ['Motor_current', 'DV_eletric'],
            'rule': 'conditional',
            'description': 'High motor current should correlate with DV_eletric=1'
        }
    ]
    
    def __init__(self):
        self.alerts = []
        self.data_quality_score = 100.0
        
    def validate_sensor_reading(self, sensor_name: str, value: float, 
                               role: str = "plant_manager") -> Dict:
        """
        Validate individual sensor reading with role-based explanation
        
        Returns: {
            'status': 'normal'|'warning'|'critical',
            'message': explanation,
            'score': quality score,
            'recommendations': list of actions
        }
        """
        if sensor_name not in self.SENSOR_RANGES:
            return {'status': 'normal', 'message': 'Sensor not in validation range', 'score': 100}
        
        ranges = self.SENSOR_RANGES[sensor_name]
        status = 'normal'
        messages = []
        recommendations = []
        score = 100
        
        # Check critical thresholds
        if value < ranges['critical_low']:
            status = 'critical'
            score = 30
            messages.append(f"CRITICAL: {sensor_name} reading {value:.2f}{ranges['unit']} is dangerously low (below {ranges['critical_low']}{ranges['unit']})")
            
            if role == "plant_manager":
                recommendations.append(f"Immediate action required: {sensor_name} critically low. Stop operations and investigate system pressure/flow.")
            elif role == "maintenance_engineer":
                recommendations.append(f"Check for leaks in {sensor_name} line, verify sensor calibration, inspect upstream components for blockages.")
            else:  # ml_engineer
                recommendations.append(f"Data anomaly detected: {sensor_name} reading outside training distribution. May indicate sensor fault or rare operational state.")
                
        elif value > ranges['critical_high']:
            status = 'critical'
            score = 30
            messages.append(f"CRITICAL: {sensor_name} reading {value:.2f}{ranges['unit']} is dangerously high (above {ranges['critical_high']}{ranges['unit']})")
            
            if role == "plant_manager":
                recommendations.append(f"Emergency: {sensor_name} at critical levels. Reduce system load immediately to prevent equipment damage.")
            elif role == "maintenance_engineer":
                recommendations.append(f"Inspect pressure relief valves, check for blockages in {sensor_name} circuit, verify cooling system if temperature-related.")
            else:  # ml_engineer
                recommendations.append(f"Outlier detected: {sensor_name} exceeds normal operating envelope. Model predictions may be less reliable in this regime.")
                
        # Check warning thresholds
        elif value < ranges['min']:
            status = 'warning'
            score = 70
            messages.append(f"WARNING: {sensor_name} reading {value:.2f}{ranges['unit']} below normal range ({ranges['min']}-{ranges['max']}{ranges['unit']})")
            
            if role == "plant_manager":
                recommendations.append(f"Monitor {sensor_name} closely. Schedule inspection within 24 hours.")
            elif role == "maintenance_engineer":
                recommendations.append(f"Investigate {sensor_name} trend. Check for minor leaks or sensor drift.")
            else:  # ml_engineer
                recommendations.append(f"{sensor_name} approaching low boundary of training data. Monitor model confidence.")
                
        elif value > ranges['max']:
            status = 'warning'
            score = 70
            messages.append(f"WARNING: {sensor_name} reading {value:.2f}{ranges['unit']} above normal range ({ranges['min']}-{ranges['max']}{ranges['unit']})")
            
            if role == "plant_manager":
                recommendations.append(f"{sensor_name} elevated. Increase monitoring frequency and prepare for maintenance.")
            elif role == "maintenance_engineer":
                recommendations.append(f"Check {sensor_name} system for early signs of blockage or excessive load.")
            else:  # ml_engineer
                recommendations.append(f"{sensor_name} in high operational zone. Feature importance may shift in predictions.")
        else:
            messages.append(f"✓ {sensor_name}: {value:.2f}{ranges['unit']} within normal range ({ranges['min']}-{ranges['max']}{ranges['unit']})")
        
        return {
            'status': status,
            'message': ' '.join(messages),
            'score': score,
            'recommendations': recommendations
        }
    
    def validate_cross_sensor_correlations(self, sensor_values: Dict[str, float], 
                                          role: str = "plant_manager") -> List[Dict]:
        """
        Validate relationships between sensors
        """
        correlation_alerts = []
        
        # Check TP3 vs Reservoirs
        if 'TP3' in sensor_values and 'Reservoirs' in sensor_values:
            diff = abs(sensor_values['TP3'] - sensor_values['Reservoirs'])
            if diff > 0.5:
                alert = {
                    'status': 'warning',
                    'message': f"Pressure mismatch: TP3 ({sensor_values['TP3']:.2f}bar) and Reservoirs ({sensor_values['Reservoirs']:.2f}bar) differ by {diff:.2f}bar",
                    'recommendations': []
                }
                
                if role == "plant_manager":
                    alert['recommendations'].append("Pressure discrepancy detected between pneumatic panel and reservoirs. Investigate for leakage or valve issues.")
                elif role == "maintenance_engineer":
                    alert['recommendations'].append("Check pneumatic lines between TP3 and reservoirs for leaks. Inspect isolation valves and pressure regulators.")
                else:
                    alert['recommendations'].append("Feature correlation deviation detected. Model may identify this as anomalous pattern.")
                    
                correlation_alerts.append(alert)
        
        # Check Motor Current vs DV_eletric
        if 'Motor_current' in sensor_values and 'DV_eletric' in sensor_values:
            if sensor_values['Motor_current'] > 6.0 and sensor_values['DV_eletric'] == 0:
                alert = {
                    'status': 'warning',
                    'message': f"Inconsistent state: High motor current ({sensor_values['Motor_current']:.2f}A) but outlet valve (DV_eletric) is closed",
                    'recommendations': []
                }
                
                if role == "plant_manager":
                    alert['recommendations'].append("System configuration error: motor running but valve closed. This indicates control system malfunction.")
                elif role == "maintenance_engineer":
                    alert['recommendations'].append("Verify DV_eletric valve operation. Check control system signals and valve actuator functionality.")
                else:
                    alert['recommendations'].append("State inconsistency: features suggest conflicting operational modes. May confuse model predictions.")
                    
                correlation_alerts.append(alert)
        
        # Check Oil Level emergency
        if sensor_values.get('Oil_level', 0) == 1:
            alert = {
                'status': 'critical',
                'message': "CRITICAL: Low oil level detected - immediate shutdown required",
                'recommendations': []
            }
            
            if role == "plant_manager":
                alert['recommendations'].append("EMERGENCY: Stop all operations immediately. Low oil will cause catastrophic compressor damage within minutes.")
            elif role == "maintenance_engineer":
                alert['recommendations'].append("URGENT: Shutdown compressor NOW. Refill oil to proper level, check for leaks, inspect oil pump operation before restart.")
            else:
                alert['recommendations'].append("Critical safety feature triggered. Model will strongly predict failure - this is expected and correct behavior.")
                
            correlation_alerts.append(alert)
        
        return correlation_alerts
    
    def calculate_data_quality_score(self, sensor_values: Dict[str, float]) -> Dict:
        """
        Calculate comprehensive data quality score based on multiple factors:
        1. Range Compliance (40%): How well values stay within normal ranges
        2. Cross-Correlation (30%): Sensor reading relationships
        3. Completeness (20%): Missing or null values
        4. Signal Stability (10%): Sudden changes or spikes
        """
        scores = {
            'range_compliance': 100,
            'cross_correlation': 100,
            'completeness': 100,
            'signal_stability': 100
        }
        
        # 1. Range Compliance (40%)
        range_violations = 0
        critical_violations = 0
        for sensor, value in sensor_values.items():
            if sensor in self.SENSOR_RANGES:
                ranges = self.SENSOR_RANGES[sensor]
                if value < ranges['critical_low'] or value > ranges['critical_high']:
                    critical_violations += 1
                elif value < ranges['min'] or value > ranges['max']:
                    range_violations += 1
        
        if critical_violations > 0:
            scores['range_compliance'] = max(0, 100 - (critical_violations * 40))
        elif range_violations > 0:
            scores['range_compliance'] = max(60, 100 - (range_violations * 20))

        # 2. Cross-Correlation (30%)
        correlation_violations = 0
        for rule in self.CORRELATION_RULES:
            if rule['rule'] == 'should_match':
                sensor1, sensor2 = rule['sensors']
                if sensor1 in sensor_values and sensor2 in sensor_values:
                    diff = abs(sensor_values[sensor1] - sensor_values[sensor2])
                    if diff > rule['tolerance']:
                        correlation_violations += 1
            elif rule['rule'] == 'conditional':
                # Check conditional relationships
                if ('Motor_current' in sensor_values and 'DV_eletric' in sensor_values and
                    sensor_values['Motor_current'] > 6.0 and sensor_values['DV_eletric'] == 0):
                    correlation_violations += 1
        
        scores['cross_correlation'] = max(0, 100 - (correlation_violations * 25))

        # 3. Completeness (20%)
        expected_sensors = set(self.SENSOR_RANGES.keys())
        provided_sensors = set(sensor_values.keys())
        missing_ratio = len(expected_sensors - provided_sensors) / len(expected_sensors)
        scores['completeness'] = max(0, 100 - (missing_ratio * 100))

        # 4. Signal Stability (10%)
        # This would ideally compare with recent historical values
        # For now, we'll check for values in extreme ranges
        extreme_values = 0
        for sensor, value in sensor_values.items():
            if sensor in self.SENSOR_RANGES:
                ranges = self.SENSOR_RANGES[sensor]
                normal_range = ranges['max'] - ranges['min']
                if value > ranges['max'] + (normal_range * 0.5):
                    extreme_values += 1
        
        scores['signal_stability'] = max(0, 100 - (extreme_values * 20))

        # Calculate weighted final score
        final_score = (
            scores['range_compliance'] * 0.4 +
            scores['cross_correlation'] * 0.3 +
            scores['completeness'] * 0.2 +
            scores['signal_stability'] * 0.1
        )

        return {
            'overall_score': final_score,
            'component_scores': scores,
            'details': {
                'range_violations': range_violations,
                'critical_violations': critical_violations,
                'correlation_violations': correlation_violations,
                'missing_sensors': list(expected_sensors - provided_sensors),
                'extreme_values': extreme_values
            }
        }

    def generate_data_quality_report(self, sensor_values: Dict[str, float], 
                                    role: str = "plant_manager") -> Dict:
        """Comprehensive data quality assessment with detailed explanations"""
        quality_score = self.calculate_data_quality_score(sensor_values)
        self.alerts = []
        quality_factors = []
        
        # Add quality factor explanations based on component scores
        if quality_score['component_scores']['range_compliance'] < 100:
            quality_factors.append({
                'sensor': 'Multiple',
                'score': quality_score['component_scores']['range_compliance'],
                'reason': f"Range violations detected: {quality_score['details']['range_violations']} warnings, {quality_score['details']['critical_violations']} critical"
            })
        
        if quality_score['component_scores']['cross_correlation'] < 100:
            quality_factors.append({
                'sensor': 'Multiple',
                'score': quality_score['component_scores']['cross_correlation'],
                'reason': f"Cross-correlation issues detected: {quality_score['details']['correlation_violations']} violations"
            })
        
        if quality_score['component_scores']['completeness'] < 100:
            quality_factors.append({
                'sensor': 'Multiple',
                'score': quality_score['component_scores']['completeness'],
                'reason': f"Incomplete data: {quality_score['details']['missing_sensors']} sensors missing"
            })
        
        if quality_score['component_scores']['signal_stability'] < 100:
            quality_factors.append({
                'sensor': 'Multiple',
                'score': quality_score['component_scores']['signal_stability'],
                'reason': f"Signal stability concerns: {quality_score['details']['extreme_values']} extreme values detected"
            })
        
        # Check individual sensor readings for additional alerts
        for sensor, value in sensor_values.items():
            if sensor in self.SENSOR_RANGES:
                ranges = self.SENSOR_RANGES[sensor]
                if value < ranges['critical_low'] or value > ranges['critical_high']:
                    self.alerts.append({
                        'status': 'critical',
                        'message': f"{sensor} reading {value:.2f}{ranges['unit']} out of critical range",
                        'recommendations': []
                    })
                elif value < ranges['min'] or value > ranges['max']:
                    self.alerts.append({
                        'status': 'warning',
                        'message': f"{sensor} reading {value:.2f}{ranges['unit']} out of normal range",
                        'recommendations': []
                    })
        
        # Sort alerts by severity
        self.alerts.sort(key=lambda x: x['status'] == 'normal')
        
        return {
            'overall_status': 'critical' if quality_score['overall_score'] < 60 
                             else 'warning' if quality_score['overall_score'] < 80 
                             else 'normal',
            'data_quality_score': quality_score['overall_score'],
            'component_scores': quality_score['component_scores'],
            'quality_factors': quality_factors,
            'alerts': self.alerts,
            'details': quality_score['details'],
            'timestamp': datetime.now().isoformat()
        }