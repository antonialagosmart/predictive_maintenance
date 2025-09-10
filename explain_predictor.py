import os
import shap
import pandas as pd
import joblib   # use joblib instead of pickle for sklearn/xgboost models
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import io

# ----------------------
# Load model and scaler
# ----------------------
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Reconstruct feature names (must match training)
feature_names = list(model.get_booster().feature_names)

# Define sensor columns that need scaling (must match training)
sensor_cols = [
    'TP2',           # Compressor pressure (continuous, unit: bar)
    'TP3',           # Pneumatic panel pressure (continuous, unit: bar)
    'H1',            # Pressure drop across cyclonic separator (continuous, unit: bar)
    'DV_pressure',   # Pressure drop at towers (continuous, unit: bar)
    'Reservoirs',    # Downstream reservoir pressure (continuous, unit: bar)
    'Motor_current', # One phase motor current (continuous, unit: A)
    'Oil_temperature'# Compressor oil temperature (continuous, unit: °C)
]

# Comprehensive sensor descriptions and operational contexts
SENSOR_DESCRIPTIONS = {
    'TP2': {
        'name': 'Compressor Pressure Sensor (TP2): ',
        'description': '',
        'normal_range': '6-10 bar',
        'critical_thresholds': {'high': 9.5, 'low': 2.0},
        'operational_context': 'Higher values indicate increased compression load, while low values suggest insufficient compression or potential leakage'
    },
    'TP3': {
        'name': 'Pneumatic Panel Pressure (TP3): ',
        'description': '',
        'normal_range': '7-11 bar',
        'critical_thresholds': {'high': 10.0, 'low': 6.0},
        'operational_context': 'Should correlate closely with reservoir pressure. Significant deviations indicate pneumatic system issues'
    },
    'H1': {
        'name': 'Cyclonic Separator Pressure Drop (H1): ',
        'description': '',
        'normal_range': '0.1-2.0 bar',
        'critical_thresholds': {'high': 1.8, 'low': 0.05},
        'operational_context': 'High values suggest filter clogging or blockage, low values may indicate bypass or damaged separator'
    },
    'DV_pressure': {
        'name': 'Air Dryer Tower Discharge Pressure (DV_pressure): ',
        'description': '',
        'normal_range': '0-2.5 bar',
        'critical_thresholds': {'high': 2.0, 'low': -0.1},
        'operational_context': 'Zero values are normal during loaded operation. Non-zero values indicate tower switching or maintenance cycles'
    },
    'Reservoirs': {
        'name': 'Downstream Reservoir Pressure (Reservoirs):',
        'description': '',
        'normal_range': '7-11 bar',
        'critical_thresholds': {'high': 10.5, 'low': 6.5},
        'operational_context': 'Should closely match TP3 pressure. Major differences indicate leakage or reservoir system faults'
    },
    'Motor_current': {
        'name': 'Three-Phase Motor Current (Motor_current):',
        'description': '',
        'normal_range': '0-9 A',
        'critical_thresholds': {'high': 8.5, 'low': 0.5},
        'operational_context': '~0A: motor off, ~4A: offloaded operation, ~7A: under load, ~9A: startup. Values outside range indicate electrical issues'
    },
    'Oil_temperature': {
        'name': 'Compressor Oil Temperature (Oil_temperature):  ',
        'description': '',
        'normal_range': '40-65°C',
        'critical_thresholds': {'high': 70.0, 'low': 25.0},
        'operational_context': 'High temperatures cause oil degradation and component wear. Low temperatures may indicate insufficient load or cooling issues'
    },
    'COMP': {
        'name': 'Air Intake Valve Signal (COMP): ',
        'description': '',
        'states': {'0': 'Valve open - air intake active', '1': 'Valve closed - no air intake'},
        'operational_context': 'Works in conjunction with MPG signal. Active state indicates compressor in standby or maintenance mode'
    },
    'DV_eletric': {
        'name': 'Compressor Outlet Valve Control Signal (DV_eletric): ',
        'description': '',
        'states': {'0': 'Valve closed - offload/off state', '1': 'Valve open - under load operation'},
        'operational_context': 'Active during loaded operation, inactive during offload or shutdown. Critical for pressure regulation'
    },
    'Towers': {
        'name': 'Air Dryer Tower Selector Signal (Towers): ',
        'description': '',
        'states': {'0': 'Tower 1 in operation', '1': 'Tower 2 in operation'},
        'operational_context': 'Alternates between towers to allow regeneration of desiccant material. Proper switching is essential for dry air output'
    },
    'MPG': {
        'name': 'Main Pressure Gauge Start Signal (MPG): ',
        'description': '',
        'states': {'0': 'Pressure adequate - no start needed', '1': 'Low pressure detected - start sequence initiated'},
        'operational_context': 'Critical safety and operational control. Activates COMP sensor and initiates loaded operation cycle'
    },
    'LPS': {
        'name': 'Low Pressure Sensor (LPS): ',
        'description': '',
        'states': {'0': 'Normal pressure maintained', '1': 'Low pressure alarm - below 7 bar'},
        'operational_context': 'Emergency safety feature preventing system damage from insufficient pressure. Triggers immediate corrective actions'
    },
    'Pressure_switch': {
        'name': 'Tower Discharge Detector (Pressure_switch): ',
        'description': '',
        'states': {'0': 'No tower discharge detected', '1': 'Tower discharge in progress'},
        'operational_context': 'Indicates proper tower regeneration cycles. Absence during expected cycles suggests tower malfunction'
    },
    'Oil_level': {
        'name': 'Compressor Oil Level Sensor (Oil_level): ',
        'description': '',
        'states': {'0': 'Oil level adequate', '1': 'Oil level below minimum threshold'},
        'operational_context': 'Low oil levels cause catastrophic bearing and seal damage. Active signal requires immediate shutdown and maintenance'
    },
    'Caudal_impulses': {
        'name': 'Airflow Pulse Counter (Caudal_impulses): ',
        'description': '',
        'states': {'0': 'No significant airflow detected', '1': 'Active airflow pulses detected'},
        'operational_context': 'Indicates actual air delivery to system. Absence during operation suggests blockage or flow meter failure'
    }
}

# SHAP explainer
explainer = shap.Explainer(model)

def get_sensor_context(sensor_name, value, shap_contribution):
    """Generate detailed contextual explanation for a sensor reading"""
    if sensor_name not in SENSOR_DESCRIPTIONS:
        return f"{sensor_name} contributed {abs(shap_contribution):.3f} to the prediction"
    
    sensor_info = SENSOR_DESCRIPTIONS[sensor_name]
    
    # Build contextual explanation
    context = f"{sensor_info['name']} "
    
    if sensor_name in sensor_cols:  # Analog sensors
        context += f"reading of {value:.2f}"
        if 'normal_range' in sensor_info:
            context += f" (normal range: {sensor_info['normal_range']})"
        
        # Assess if value is concerning
        if 'critical_thresholds' in sensor_info:
            thresholds = sensor_info['critical_thresholds']
            if value > thresholds['high']:
                context += f" - CRITICALLY HIGH (above {thresholds['high']})"
            elif value < thresholds['low']:
                context += f" - CRITICALLY LOW (below {thresholds['low']})"
            elif value > thresholds['high'] * 0.9:
                context += f" - ELEVATED (approaching {thresholds['high']})"
            elif value < thresholds['low'] * 1.1:
                context += f" - CONCERNING (approaching {thresholds['low']})"
        
        context += f". {sensor_info['operational_context']}"
        
    else:  # Digital sensors
        state = int(value)
        if 'states' in sensor_info and str(state) in sensor_info['states']:
            context += f" {sensor_info['states'][str(state)]}"
        else:
            context += f"state: {'ACTIVE' if state else 'INACTIVE'}"
        
        context += f". {sensor_info['operational_context']}"
    
    return context

def generate_detailed_report(prediction_result, sensor_values, timestamp=None):
    """Generate a comprehensive report for download"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pred_label = prediction_result['pred_label']
    probability = float(prediction_result['probability'][0])
    explanation_text = prediction_result['explanation_text']
    
    report = f"""
METRO TRAIN APU HEALTH MONITORING REPORT
========================================

Generated: {timestamp}
Report ID: APU-{datetime.now().strftime('%Y%m%d-%H%M%S')}

EXECUTIVE SUMMARY
================
System Status: {pred_label}
Confidence Level: {probability:.1%}
Assessment: {'CRITICAL ATTENTION REQUIRED' if pred_label == 'FAILURE' else 'SYSTEM OPERATING NORMALLY'}

SENSOR READINGS AT TIME OF ANALYSIS
==================================

Analog Sensors:
--------------
• TP2 (Compressor Pressure): {sensor_values.get('TP2', 0):.3f} bar
• TP3 (Pneumatic Panel Pressure): {sensor_values.get('TP3', 0):.3f} bar
• H1 (Separator Pressure Drop): {sensor_values.get('H1', 0):.3f} bar
• DV Pressure (Tower Discharge): {sensor_values.get('DV_pressure', 0):.3f} bar
• Reservoirs (Downstream Pressure): {sensor_values.get('Reservoirs', 0):.3f} bar
• Oil Temperature: {sensor_values.get('Oil_temperature', 0):.1f}°C
• Motor Current: {sensor_values.get('Motor_current', 0):.2f}A

Digital Sensor States:
---------------------
• COMP (Air Intake Valve): {'ACTIVE' if sensor_values.get('COMP', 0) else 'INACTIVE'}
• DV Electric (Outlet Valve): {'ACTIVE' if sensor_values.get('DV_eletric', 0) else 'INACTIVE'}
• Towers (Tower Selector): {'Tower 2' if sensor_values.get('Towers', 0) else 'Tower 1'}
• MPG (Start Signal): {'ACTIVE' if sensor_values.get('MPG', 0) else 'INACTIVE'}
• LPS (Low Pressure Sensor): {'TRIGGERED' if sensor_values.get('LPS', 0) else 'NORMAL'}
• Pressure Switch: {'ACTIVE' if sensor_values.get('Pressure_switch', 0) else 'INACTIVE'}
• Oil Level Sensor: {'LOW LEVEL' if sensor_values.get('Oil_level', 0) else 'NORMAL'}
• Caudal Impulses: {'FLOW DETECTED' if sensor_values.get('Caudal_impulses', 0) else 'NO FLOW'}

DETAILED AI ANALYSIS
===================
{explanation_text}

RECOMMENDED ACTIONS
==================
{'IMMEDIATE SHUTDOWN AND MAINTENANCE REQUIRED' if pred_label == 'FAILURE' else 'CONTINUE NORMAL OPERATIONS WITH REGULAR MONITORING'}

Report generated by APU Health Monitor v2.0
Powered by Explainable AI & Advanced Machine Learning
Copyright © 2025 Odubiyi Ifeoluwa Antonia
"""
    return report

def predict_with_explanation(input_data: pd.DataFrame, output_dir="output"):
    """
    Make predictions with detailed SHAP explanations on new data.
    
    Args:
        input_data: DataFrame with raw (unscaled) sensor data
        output_dir: Directory to save plots
    
    Returns:
        predictions, pred_probs, explanations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy to avoid modifying the original data
    processed_data = input_data.copy()
    
    # Drop unwanted columns if present
    if 'Unnamed: 0' in processed_data.columns:
        processed_data.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Apply scaling to sensor columns (same as training)
    processed_data[sensor_cols] = scaler.transform(processed_data[sensor_cols])
    
    # Ensure input data columns align with training features
    processed_data = processed_data[feature_names]
    
    predictions = model.predict(processed_data)
    proba_all = model.predict_proba(processed_data)  # [[p0, p1], ...]

    shap_values_all = explainer(processed_data)
    explanations = []

    for i in range(len(processed_data)):
        shap_exp = shap_values_all[i]
        shap_contribs = np.array(shap_exp.values).flatten()
        feature_vals = np.array(shap_exp.data).flatten()  # These are scaled values
        base_value = shap_exp.base_values

        # Compute f(x) and Ef(x) in probability terms
        fx_logodds = shap_contribs.sum() + base_value
        fx_prob = expit(fx_logodds)
        base_prob = expit(base_value)

        # Top 5 contributing features
        top_indices = np.argsort(np.abs(shap_contribs))[-5:][::-1]
        top_shap_values = shap_contribs[top_indices]
        top_feature_values = feature_vals[top_indices]  # Scaled values
        top_feature_names = [feature_names[j] for j in top_indices]
        
        # Get original (unscaled) values for display
        original_values = []
        for j in top_indices:
            feature_name = feature_names[j]
            if feature_name in sensor_cols:
                # Inverse transform to get original value
                scaled_val = feature_vals[j]
                # Create array with proper shape for inverse_transform
                sensor_idx = sensor_cols.index(feature_name)
                temp_array = np.zeros((1, len(sensor_cols)))
                temp_array[0, sensor_idx] = scaled_val
                original_val = scaler.inverse_transform(temp_array)[0, sensor_idx]
                original_values.append(original_val)
            else:
                # Digital sensors remain as-is
                original_values.append(feature_vals[j])

        # ---------------------------
        # Pick label + probability
        # ---------------------------
        failure_prob = proba_all[i, 1]
        normal_prob = proba_all[i, 0]

        if predictions[i] == 1:  # FAILURE
            pred_label = "FAILURE"
            pred_prob = failure_prob
        else:  # NORMAL
            pred_label = "NORMAL"
            pred_prob = normal_prob

        # Start building comprehensive explanation
        explanation = (
            f"Row {i}: The AI model predicts {pred_label} "
            f"with confidence of {pred_prob:.1%}.\n\n"
            # f"(model probability: {fx_prob:.3f}, baseline: {base_prob:.3f}).\n\n"
        )

        # Enhanced warning system
        if pred_label == "NORMAL" and failure_prob >= 0.40:
            explanation += (
                f"ATTENTION REQUIRED: Although predicted as NORMAL, the failure probability is {failure_prob:.1%}. "
                f"The system appears to be transitioning toward a potential failure state. Increased monitoring is recommended.\n\n"
            )
        elif pred_label == "FAILURE" and failure_prob >= 0.90:
            explanation += (
                f"CRITICAL ALERT: Very high failure probability ({failure_prob:.1%}). "
                f"Immediate intervention is strongly recommended to prevent system failure.\n\n"
            )

        explanation += (
            "DETAILED SENSOR ANALYSIS\n\n"
            "The following sensor readings and system states contributed most significantly to this prediction:\n\n"
        )

        # ---------------------------
        # Detailed sensor explanations - FIXED FORMATTING
        # ---------------------------
        recommendations = []
        critical_issues = []
        
        for rank, (fname, sval, orig_val) in enumerate(zip(top_feature_names, top_shap_values, original_values), 1):
            contribution_direction = "PUSHING" if sval > 0 else "PUSHING"
            contribution_magnitude = "strongly" if abs(sval) > 0.1 else "moderately" if abs(sval) > 0.05 else "slightly"
            
            # Get detailed sensor context
            sensor_context = get_sensor_context(fname, orig_val, sval)
            
            # FIXED: Single formatted explanation without double bullets
            explanation += (
                f"{rank}. {sensor_context}\n"
                f"Impact: This reading is {contribution_magnitude} {contribution_direction} "
                f"the model's confidence toward {'FAILURE' if sval > 0 else 'NORMAL OPERATION'} "
                f"by {abs(sval):.3f} units.\n\n"
            )

           # Generate specific recommendations and identify critical issues
            if pred_label == "FAILURE":
                if sval > 0:  # Contributing to failure
                    if fname in ['TP2', 'TP3'] and orig_val > SENSOR_DESCRIPTIONS[fname]['critical_thresholds']['high']:
                        critical_issues.append(
                            f"Critical pressure overload in {SENSOR_DESCRIPTIONS[fname]['name']}"
                        )
                        recommendations.append(
                            f"⚠️ URGENT: {SENSOR_DESCRIPTIONS[fname]['name']} is above safe limits "
                            f"(current: {orig_val}, threshold: {SENSOR_DESCRIPTIONS[fname]['critical_thresholds']['high']}). "
                            f"Reduce system pressure immediately to prevent seal damage, leakage, or catastrophic failure."
                        )

                    elif fname == 'Motor_current' and orig_val > 8.0:
                        critical_issues.append("Motor electrical overload detected")
                        recommendations.append(
                            f"⚡ CRITICAL: Motor current is {orig_val}A (limit: 8A). "
                            f"This indicates potential electrical overload or excessive mechanical demand. "
                            f"Inspect for binding, over-compression, or electrical short circuits immediately."
                        )

                    elif fname == 'Oil_temperature' and orig_val > SENSOR_DESCRIPTIONS[fname]['critical_thresholds']['high']:
                        critical_issues.append("Compressor oil overheating")
                        recommendations.append(
                            f"🔥 IMMEDIATE ACTION: Oil temperature at {orig_val}°C exceeds safe limit "
                            f"({SENSOR_DESCRIPTIONS[fname]['critical_thresholds']['high']}°C). "
                            f"Shut down compressor, check cooling system efficiency, verify oil quality/level, "
                            f"and inspect for heat exchanger blockages."
                        )

                    elif fname in ['LPS', 'Oil_level'] and orig_val == 1:
                        critical_issues.append(f"Safety system activation: {SENSOR_DESCRIPTIONS[fname]['name']}")
                        recommendations.append(
                            f"🚨 SAFETY ALERT: {SENSOR_DESCRIPTIONS[fname]['name']} has triggered. "
                            f"{SENSOR_DESCRIPTIONS[fname]['operational_context']} "
                            f"Stop system and investigate root cause immediately."
                        )

                    else:
                        recommendations.append(
                            f"⚠️ WARNING: {SENSOR_DESCRIPTIONS[fname]['name']} is trending toward unsafe operation "
                            f"(current: {orig_val}). Proactive adjustment recommended."
                        )

                else:  # Helping prevent failure (negative contribution)
                    recommendations.append(
                        f"✅ MAINTAIN: {SENSOR_DESCRIPTIONS[fname]['name']} is stabilizing the system "
                        f"(current: {orig_val}). Continue monitoring to ensure it stays within optimal range."
                    )

            else:  # NORMAL prediction
                recommendations.append(
                    f"✅ NORMAL: {SENSOR_DESCRIPTIONS[fname]['name']} is within safe range "
                    f"(current: {orig_val}). Maintain current operating conditions."
                )

                if sval > 0 and abs(sval) > 0.05:  # Contributing to normal operation
                    recommendations.append(
                        f"OPTIMAL: {fname} reading ({orig_val:.2f}) is supporting healthy system operation. "
                        f"Continue current operational parameters."
                    )
                elif sval < 0 and abs(sval) > 0.05:  # Could lead to issues
                    if fname in sensor_cols and orig_val < SENSOR_DESCRIPTIONS[fname]['critical_thresholds']['low'] * 1.2:
                        recommendations.append(
                            f"MONITOR: {fname} value ({orig_val:.2f}) is trending toward concerning levels. "
                            f"Increase monitoring frequency and investigate potential causes."
                        )

        # Add critical issues summary
        if critical_issues:
            explanation += f"CRITICAL ISSUES IDENTIFIED\n"
            for issue in critical_issues:
                explanation += f"{issue}\n"
            explanation += "\n"

        # Most influential factor summary
        most_influential = top_feature_names[0]
        most_influential_impact = "driving the system toward failure" if top_shap_values[0] > 0 else "helping maintain normal operation"
        explanation += (
            f"PRIMARY FACTOR\n"
            f"{SENSOR_DESCRIPTIONS[most_influential]['name']} is the dominant factor {most_influential_impact}. "
            f"Its current state/value ({original_values[0]:.2f}) has the strongest influence on the prediction outcome.\n\n"
        )

        # Recommendations section
        if recommendations:
            explanation += "MAINTENANCE RECOMMENDATIONS\n"
            for rec in recommendations:
                explanation += f"{rec}\n"
        else:
            explanation += "SYSTEM STATUS\nNo specific maintenance actions required at this time. Continue normal monitoring procedures.\n"

        # Enhanced horizontal bar plot with better styling
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        colors = ["#e74c3c" if val > 0 else "#27ae60" for val in top_shap_values]  # Red for failure, green for normal
        
        bars = ax.barh(top_feature_names[::-1], top_shap_values[::-1], color=colors[::-1], alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_shap_values[::-1])):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=7)
        
        ax.set_xlabel("SHAP Impact Value\n(← Normal    Failure →)", fontsize=9)
        ax.set_title("AI Model: Top 5 Most Influential Factors", fontsize=10, pad=15)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(pad=1.0)

        plot_path = os.path.join(output_dir, f"shap_detailed_row_{i}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close(fig)

        explanations.append((explanation, plot_path))

    # Return predictions + probability of predicted class
    pred_probs = [proba_all[j, predictions[j]] for j in range(len(predictions))]
    return predictions, pred_probs, explanations