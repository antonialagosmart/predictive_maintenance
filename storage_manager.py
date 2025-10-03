"""
Local Storage Manager
Handles saving predictions to JSON and provides trend analysis capabilities
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

class StorageManager:
    """Manage local JSON storage for predictions and trend analysis"""
    
    # Add sensor ranges definition at class level
    SENSOR_RANGES = {
        'TP2': {'min': 6.0, 'max': 10.0, 'critical_low': 2.0, 'critical_high': 9.5, 'unit': 'bar', 
                'description': 'Higher values indicate increased compression load, while low values suggest insufficient compression or potential leakage'},
        'TP3': {'min': 7.0, 'max': 11.0, 'critical_low': 6.0, 'critical_high': 10.0, 'unit': 'bar',
                'description': 'Should correlate closely with reservoir pressure. Significant deviations indicate pneumatic system issues'},
        'H1': {'min': 0.1, 'max': 2.0, 'critical_low': 0.05, 'critical_high': 1.8, 'unit': 'bar',
               'description': 'High values suggest filter clogging or blockage, low values may indicate bypass or damaged separator'},
        'DV_pressure': {'min': 0.0, 'max': 2.5, 'critical_low': -0.1, 'critical_high': 2.0, 'unit': 'bar',
                       'description': 'Zero values are normal during loaded operation. Non-zero values indicate tower switching or maintenance cycles'},
        'Reservoirs': {'min': 7.0, 'max': 11.0, 'critical_low': 6.5, 'critical_high': 10.5, 'unit': 'bar',
                      'description': 'Should closely match TP3 pressure. Major differences indicate leakage or reservoir system faults'},
        'Motor_current': {'min': 0.0, 'max': 9.0, 'critical_low': 0.5, 'critical_high': 8.5, 'unit': 'A',
                         'description': '~0A: motor off, ~4A: offloaded operation, ~7A: under load, ~9A: startup'},
        'Oil_temperature': {'min': 40.0, 'max': 65.0, 'critical_low': 25.0, 'critical_high': 70.0, 'unit': '°C',
                           'description': 'High temperatures cause oil degradation and component wear. Low temperatures may indicate insufficient load'}
    }

    def __init__(self, storage_file: str = "apu_prediction_history.json"):
        self.storage_file = storage_file
        self.predictions = []
        self.load_history()
    
    def load_history(self):
        """Load existing prediction history from JSON file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get('predictions', [])
            except Exception as e:
                print(f"Error loading history: {e}")
                self.predictions = []
        else:
            self.predictions = []
    
    def save_prediction(self, prediction_data: Dict) -> bool:
        """Enhanced prediction saving with better error handling"""
        try:
            # Ensure timestamp exists
            if 'timestamp' not in prediction_data:
                prediction_data['timestamp'] = datetime.now().isoformat()
            
            # Validate required fields
            required_fields = ['prediction', 'probability', 'sensor_values']
            for field in required_fields:
                if field not in prediction_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Add to predictions list
            self.predictions.append(prediction_data)
            
            # Save to file with error handling
            try:
                with open(self.storage_file, 'w') as f:
                    json.dump({
                        'predictions': self.predictions,
                        'last_updated': datetime.now().isoformat()
                    }, f, indent=2)
                return True
            except Exception as e:
                print(f"Error saving to file: {e}")
                return False
                
        except Exception as e:
            print(f"Error processing prediction: {e}")
            return False
    
    def get_all_predictions(self) -> List[Dict]:
        """Retrieve all stored predictions"""
        return self.predictions
    
    def get_recent_predictions(self, hours: int = 24) -> List[Dict]:
        """Get predictions from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            p for p in self.predictions 
            if datetime.fromisoformat(p['timestamp']) >= cutoff
        ]
    
    def get_trend_analysis(self, role: str = "plant_manager", 
                      start_date=None, end_date=None, 
                      parameters=None, time_analysis="Daily Overview",
                      start_time=None, end_time=None) -> Dict:
        """Enhanced trend analysis with filtering and parameter comparison"""
        if not self.predictions:
            return {'status': 'no_data'}
        
        # Convert predictions to DataFrame with proper timestamp handling
        df = pd.DataFrame(self.predictions)
        try:
            # Try parsing as ISO format first
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        except Exception:
            try:
                # Try parsing as custom format
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M')
            except Exception:
                # Fallback to mixed format
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        
        # Apply date filters
        if start_date:
            df = df[df['timestamp'].dt.date >= start_date]
        if end_date:
            df = df[df['timestamp'].dt.date <= end_date]
        
        # Apply time filters for within-day analysis
        if time_analysis == "Within Day Analysis" and start_time and end_time:
            df = df[
                (df['timestamp'].dt.time >= start_time) &
                (df['timestamp'].dt.time <= end_time)
            ]
        
        if len(df) == 0:
            return {'status': 'no_data'}
        
        # Extract parameters from sensor_values
        param_data = pd.DataFrame()
        for param in (parameters or []):
            param_data[param] = df['sensor_values'].apply(lambda x: x.get(param))
        
        # Generate explanations based on trends
        explanations = []
        if len(df) > 1:
            # Analyze failure trends
            recent_failure_rate = df['prediction'].tail(10).mean()
            overall_failure_rate = df['prediction'].mean()
            
            if recent_failure_rate > overall_failure_rate:
                explanations.append(f"CRITICAL: System reliability deteriorating - Recent failure rate increased to {recent_failure_rate:.1%}")
            
            # Analyze parameter trends
            for param in param_data.columns:
                recent_mean = param_data[param].tail(10).mean()
                overall_mean = param_data[param].mean()
                change_pct = ((recent_mean - overall_mean) / overall_mean) * 100
                
                if abs(change_pct) > 10:
                    direction = "increased" if change_pct > 0 else "decreased"
                    explanations.append(f"WARNING: {param} has {direction} by {abs(change_pct):.1f}% from historical average")
        
        # Create detailed trend report
        detailed_analysis = self._generate_trend_report(df, param_data, role)
        
        return {
            'status': 'success',
            'total_predictions': len(df),
            'failure_count': int(df['prediction'].sum()),
            'failure_rate': float(df['prediction'].mean()),
            'trend_direction': 'deteriorating' if df['prediction'].tail(50).mean() > df['prediction'].head(50).mean() else 'improving',
            'parameter_plot': self._create_parameter_plot(df, param_data),
            'correlation_plot': self._create_correlation_plot(param_data),
            'role_specific_insights': explanations,
            'detailed_analysis': detailed_analysis
        }

    def _generate_trend_report(self, df: pd.DataFrame, param_data: pd.DataFrame, role: str) -> Dict:
        """Generate detailed trend analysis report"""
        report = {
            'summary': [],
            'recommendations': [],
            'critical_issues': [],
            'detailed_analysis': []
        }
        
        # Analyze trends for each parameter
        for param in param_data.columns:
            if param not in self.SENSOR_RANGES:
                continue
                
            current = param_data[param].iloc[-1]
            mean = param_data[param].mean()
            std = param_data[param].std()
            max_val = param_data[param].max()
            min_val = param_data[param].min()
            
            ranges = self.SENSOR_RANGES[param]
            
            analysis = {
                'parameter': param,
                'current_value': float(current),
                'mean': float(mean),
                'std': float(std),
                'max': float(max_val),
                'min': float(min_val),
                'unit': ranges['unit'],
                'description': ranges['description']
            }
            
            # Check thresholds
            if current > ranges['critical_high'] * 0.9:
                threshold_pct = (current / ranges['critical_high']) * 100
                issue = (f"{param} at {threshold_pct:.1f}% of critical threshold "
                        f"({current:.2f} vs limit {ranges['critical_high']} {ranges['unit']})")
                report['critical_issues'].append(issue)
                
                if role == "plant_manager":
                    report['recommendations'].append(
                        f"URGENT: {param} approaching critical high limit. Current: {current:.2f} {ranges['unit']}"
                    )
            elif current < ranges['critical_low'] * 1.1:
                threshold_pct = (current / ranges['critical_low']) * 100
                issue = (f"{param} at {threshold_pct:.1f}% of minimum threshold "
                        f"({current:.2f} vs limit {ranges['critical_low']} {ranges['unit']})")
                report['critical_issues'].append(issue)
                
                if role == "maintenance_engineer":
                    report['recommendations'].append(
                        f"Technical Alert: {param} near critical low threshold. "
                        f"Inspect and calibrate sensor."
                    )
            
            # Add statistical analysis
            if abs(current - mean) > 2 * std:
                deviation_msg = (f"{param} showing significant deviation: "
                               f"current {current:.2f} vs average {mean:.2f} {ranges['unit']}")
                report['summary'].append(deviation_msg)
                
                if role == "ml_engineer":
                    report['recommendations'].append(
                        f"Statistical Anomaly: {param} outside 2σ bounds. Check for concept drift."
                    )
            
            report['detailed_analysis'].append(analysis)
        
        # Add role-specific insights
        if role == "plant_manager" and report['critical_issues']:
            report['recommendations'].append(
                "URGENT: Multiple parameters approaching critical thresholds. "
                "Schedule immediate maintenance."
            )
        elif role == "maintenance_engineer":
            for issue in report['critical_issues']:
                report['recommendations'].append(
                    f"Technical Alert: {issue} - Perform detailed diagnostic and calibration check."
                )
        elif role == "ml_engineer" and len(report['critical_issues']) > 2:
            report['recommendations'].append(
                "Multiple parameters showing anomalous behavior. Consider model retraining with recent data."
            )
        
        return report

    def _create_correlation_plot(self, param_data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap plot"""
        corr = param_data.corr()
        fig = px.imshow(
            corr,
            title="Parameter Correlations",
            labels=dict(x="Parameters", y="Parameters", color="Correlation"),
            aspect="auto",
            color_continuous_scale="RdBu_r",  # Red to Blue colorscale
            template="plotly_dark"
        )
        
        # Update layout for better readability
        fig.update_layout(
            width=700,
            height=500,
            title_x=0.5,
            xaxis_tickangle=-45,  # Rotate x-axis labels
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate="Parameter 1: %{x}<br>Parameter 2: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
        )
        
        return fig

    def _create_parameter_plot(self, df: pd.DataFrame, param_data: pd.DataFrame) -> go.Figure:
        """Create parameter trend plot"""
        fig = go.Figure()
        
        # Create traces for each parameter
        for param in param_data.columns:
            # Get parameter ranges for reference lines
            range_info = self.SENSOR_RANGES.get(param, {})
            
            # Add main parameter trace
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=param_data[param],
                name=param,
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
            # Add reference lines if ranges exist
            if range_info:
                if 'critical_high' in range_info:
                    fig.add_hline(
                        y=range_info['critical_high'],
                        line_dash="dash",
                        line_color="red",
                        opacity=0.3,
                        name=f"{param} Critical High"
                    )
                if 'critical_low' in range_info:
                    fig.add_hline(
                        y=range_info['critical_low'],
                        line_dash="dash",
                        line_color="orange",
                        opacity=0.3,
                        name=f"{param} Critical Low"
                    )
        
        # Update layout
        fig.update_layout(
            title="Parameter Trends Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            hovermode='x unified'
        )
        
        # Add range selector
        fig.update_xaxes(rangeslider_visible=True)
        
        return fig