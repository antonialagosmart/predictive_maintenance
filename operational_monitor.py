"""
Operational Explainability Module
Tracks system performance, detects concept drift, and monitors model health
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class OperationalMonitor:
    """Monitor system performance and detect degradation over time"""
    
    def __init__(self):
        self.performance_history = []
        self.prediction_history = []
        self.baseline_metrics = {
            'accuracy': 0.95,  # Expected model accuracy
            'precision': 0.93,
            'recall': 0.92
        }
        
    def log_prediction(self, prediction_data: Dict):
        """Log prediction for operational monitoring"""
        try:
            # Ensure the prediction data has required fields
            required_fields = ['prediction', 'probability', 'sensor_values']
            if not all(field in prediction_data for field in required_fields):
                print("Warning: Missing required fields in prediction data")
                return False

            # Handle probability that could be list or float
            prob = prediction_data['probability']
            if isinstance(prob, (list, np.ndarray)):
                prob = float(prob[0])
            else:
                prob = float(prob)

            # Convert numpy types to Python native types for serialization
            cleaned_data = {
                'timestamp': prediction_data.get('timestamp', datetime.now().isoformat()),
                'prediction': int(prediction_data['prediction']),
                'probability': prob,  # Now storing as single float
                'sensor_values': prediction_data['sensor_values'],
                'data_quality_score': float(prediction_data.get('data_quality_score', 100.0))
            }
            
            self.prediction_history.append(cleaned_data)
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            return True
            
        except Exception as e:
            print(f"Error in log_prediction: {str(e)}")
            return False

    def detect_concept_drift(self, role: str = "plant_manager") -> Dict:
        """
        Analyze prediction patterns to detect concept drift
        """
        if len(self.prediction_history) < 50:
            return {
                'drift_detected': False,
                'message': 'Insufficient data for drift detection (need at least 50 predictions)',
                'recommendations': []
            }
        
        recent = self.prediction_history[-50:]
        older = self.prediction_history[-100:-50] if len(self.prediction_history) >= 100 else []
        
        # Calculate failure rate in recent vs older predictions
        recent_failure_rate = sum(1 for p in recent if p['prediction'] == 1) / len(recent)
        
        drift_detected = False
        severity = 'normal'
        messages = []
        recommendations = []
        
        if older:
            older_failure_rate = sum(1 for p in older if p['prediction'] == 1) / len(older)
            rate_change = abs(recent_failure_rate - older_failure_rate)
            
            if rate_change > 0.3:  # 30% change in failure rate
                drift_detected = True
                severity = 'critical' if rate_change > 0.5 else 'warning'
                messages.append(f"Significant shift in failure prediction rate detected: {rate_change*100:.1f}% change")
                
                if role == "plant_manager":
                    recommendations.append(f"Operating conditions have changed significantly. Review recent maintenance activities and operational parameters.")
                    if recent_failure_rate > older_failure_rate:
                        recommendations.append("Failure predictions increasing - equipment may be degrading. Schedule comprehensive inspection.")
                    else:
                        recommendations.append("Failure predictions decreasing - verify if recent maintenance/adjustments were effective.")
                        
                elif role == "maintenance_engineer":
                    recommendations.append("Pattern shift detected in failure predictions. Compare current sensor readings to historical baselines.")
                    recommendations.append("Document any recent repairs, part replacements, or configuration changes that may explain this shift.")
                    
                else:  # ml_engineer
                    recommendations.append(f"Concept drift detected: prediction distribution has shifted by {rate_change*100:.1f}%.")
                    recommendations.append("Model may need retraining if this represents genuine operational regime change vs. temporary anomaly.")
                    recommendations.append("Investigate if feature distributions have shifted outside training envelope.")
        
        # Check confidence trends
        recent_confidences = [p['probability'][0] for p in recent]
        avg_confidence = np.mean(recent_confidences)
        
        if avg_confidence < 0.7:
            messages.append(f"Low prediction confidence detected (avg: {avg_confidence:.1%})")
            
            if role == "plant_manager":
                recommendations.append("System operating in uncertain conditions. Increase monitoring frequency until patterns stabilize.")
            elif role == "maintenance_engineer":
                recommendations.append("Sensor readings may be ambiguous or transitional. Verify all sensors are functioning correctly.")
            else:
                recommendations.append("Model uncertainty elevated. System may be in state not well-represented in training data.")
        
        # Check data quality trends
        recent_dq_scores = [p.get('data_quality_score', 100) for p in recent]
        avg_dq_score = np.mean(recent_dq_scores)
        
        if avg_dq_score < 80:
            drift_detected = True
            messages.append(f"Data quality degradation detected (avg score: {avg_dq_score:.1f}/100)")
            
            if role == "plant_manager":
                recommendations.append("Multiple sensor anomalies detected. System reliability compromised - schedule immediate diagnostic check.")
            elif role == "maintenance_engineer":
                recommendations.append("Sensor performance declining. Calibrate all sensors and check for physical damage or environmental issues.")
            else:
                recommendations.append("Input data quality declining. Model predictions may be unreliable. Prioritize sensor maintenance.")
        
        return {
            'drift_detected': drift_detected,
            'severity': severity,
            'recent_failure_rate': recent_failure_rate,
            'avg_confidence': avg_confidence,
            'avg_data_quality': avg_dq_score,
            'messages': messages,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_system_health_report(self, role: str = "plant_manager") -> Dict:
        """Comprehensive system health assessment"""
        try:
            if not self.prediction_history:
                return {
                    'status': 'unknown',
                    'summary': ["No prediction history available. Make some predictions to generate health report."],
                    'metrics': {},
                    'timestamp': datetime.now().isoformat()
                }

            # Get recent predictions (last 100 or all if less)
            recent_predictions = self.prediction_history[-100:]
            
            # Calculate metrics safely - now probability is already a float
            failure_rate = sum(1 for p in recent_predictions if p['prediction'] == 1) / len(recent_predictions)
            avg_confidence = np.mean([p['probability'] for p in recent_predictions])
            avg_data_quality = np.mean([p.get('data_quality_score', 100.0) for p in recent_predictions])
            
            # Print debug info
            print(f"Debug - Predictions found: {len(recent_predictions)}")
            print(f"Debug - Failure rate: {failure_rate}")
            print(f"Debug - Avg confidence: {avg_confidence}")
            print(f"Debug - Avg data quality: {avg_data_quality}")
            
            # Determine health status with more granular thresholds
            if failure_rate > 0.3 or avg_data_quality < 70:
                health_status = 'poor'
            elif failure_rate > 0.1 or avg_data_quality < 85:
                health_status = 'fair'
            else:
                health_status = 'good'

            metrics = {
                'failure_rate': float(failure_rate),
                'avg_confidence': float(avg_confidence),
                'avg_data_quality': float(avg_data_quality),
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_predictions)
            }

            summary = []
            summary.append(f"System Health Status: {health_status.upper()}")
            summary.append(f"Based on {len(recent_predictions)} recent predictions")
            summary.append(f"Failure Rate: {failure_rate*100:.1f}%")
            summary.append(f"Average Confidence: {avg_confidence*100:.1f}%")
            summary.append(f"Data Quality Score: {avg_data_quality:.1f}/100")

            if health_status == 'poor':
                summary.append("⚠️ URGENT: System showing critical issues - Immediate attention required")
            elif health_status == 'fair':
                summary.append("⚠️ CAUTION: System showing early warning signs - Schedule maintenance soon")
            else:
                summary.append("✅ System operating within normal parameters")

            return {
                'status': health_status,
                'summary': summary,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error in health report generation: {str(e)}")
            return {
                'status': 'error',
                'summary': [f"Error generating health report: {str(e)}"],
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }

    def get_trend_statistics(self) -> Dict:
        """
        Calculate statistical trends for display
        """
        if not self.prediction_history:
            return {}
        
        df = pd.DataFrame(self.prediction_history)
        
        # Time-based analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate rolling statistics
        df['failure'] = df['prediction']
        df['rolling_failure_rate'] = df['failure'].rolling(window=10, min_periods=1).mean()
        
        return {
            'total_predictions': len(df),
            'total_failures': int(df['failure'].sum()),
            'overall_failure_rate': float(df['failure'].mean()),
            'recent_10_failure_rate': float(df['failure'].tail(10).mean()),
            'trend_data': df[['timestamp', 'failure', 'rolling_failure_rate']].to_dict('records')
        }