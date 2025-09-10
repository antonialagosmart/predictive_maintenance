import streamlit as st 
import pandas as pd
import os
from datetime import datetime
from explain_predictor import predict_with_explanation, generate_detailed_report

# =========================
# Page config with enhanced CSS
# =========================
st.set_page_config(
    page_title="APU Health Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Dark Mode CSS for premium look
dark_premium_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #4d4d4d 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

.main {
    background: transparent;
}

.block-container {
    padding: 2rem 1rem;
    max-width: 1400px;
}

/* Header Styling */
.main-header {
    background: rgba(15, 15, 35, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    z-index: -1;
}

.main-header h1 {
    background: linear-gradient(135deg, #667eea, #764ba2, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    font-size: 2.8rem;
    font-weight: 700;
    text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
}

.main-header .subtitle {
    color: #a0a0b0;
    margin-top: 0.8rem;
    font-size: 1.2rem;
    font-weight: 400;
    opacity: 0.9;
}

/* Card Styling */
.card {
    background: rgba(20, 20, 40, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(102, 126, 234, 0.3);
}

/* Panel Styling */
.sensor-panel {
    background: rgba(15, 15, 35, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2.5rem;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    height: fit-content;
}

.results-panel {
    background: rgba(15, 15, 35, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2.5rem;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    min-height: 500px;
}

/* Status Cards */
.status-normal {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: white;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid rgba(0, 184, 148, 0.3);
}

.status-failure {
    background: linear-gradient(135deg, #e17055, #d63031);
    color: white;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid rgba(225, 112, 85, 0.3);
}

.status-waiting {
    background: linear-gradient(135deg, #fdcb6e, #e17055);
    color: white;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid rgba(253, 203, 110, 0.3);
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 1.2rem 3rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4) !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stButton > button:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.6) !important;
    background: linear-gradient(135deg, #764ba2, #667eea) !important;
}

/* Toggle Switch Styling */
.stCheckbox > label {
    background: rgba(30, 30, 60, 0.8) !important;
    border-radius: 25px !important;
    padding: 0.8rem 1.5rem !important;
    margin: 0.3rem 0 !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    color: #e0e0e0 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

.stCheckbox > label:hover {
    background: rgba(40, 40, 70, 0.9) !important;
    border: 1px solid rgba(102, 126, 234, 0.5) !important;
    transform: translateY(-2px) !important;
}

.stCheckbox input[type="checkbox"]:checked + label {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: 1px solid rgba(102, 126, 234, 0.8) !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
}

/* Input Toggle Styling */
.input-toggle {
    background: rgba(30, 30, 60, 0.8);
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(102, 126, 234, 0.3);
}

.input-toggle .stRadio > div {
    flex-direction: row !important;
    gap: 2rem !important;
}

.input-toggle .stRadio label {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}

/* Slider Styling */
.stSlider > div > div > div {
    height: 6px !important;
}

.stSlider > div > div > div > div {
    background: white !important;
    box-shadow: 0 0 15px rgba(102, 126, 234, 0.5) !important;
    width: 20px !important;
    height: 20px !important;
}

.stSlider label {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}

/* Number Input Styling */
.stNumberInput > div > div > input {
    background: rgba(30, 30, 60, 0.8) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    padding: 0.8rem !important;
}

.stNumberInput label {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}

/* Expander Styling */
.streamlit-expanderHeader {
    background: rgba(102, 126, 234, 0.2) !important;
    border-radius: 15px !important;
    font-weight: 600 !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
}

.streamlit-expanderContent {
    background: rgba(20, 20, 40, 0.9) !important;
    border-radius: 0 0 15px 15px !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-top: none !important;
}

/* Legend Styling - More Compact */
.legend-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.4rem;
    font-size: 0.85rem;
}

.legend-item {
    background: rgba(30, 30, 60, 0.6);
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    border: 1px solid rgba(102, 126, 234, 0.2);
    color: #e0e0e0;
}

.legend-item strong {
    color: #667eea;
}

/* Sensor Section Styling */
.sensor-section {
    background: rgba(25, 25, 50, 0.8);
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(102, 126, 234, 0.2);
}

.sensor-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

/* Metric Cards */
.metric-card {
    background: rgba(30, 30, 60, 0.9);
    border-radius: 18px;
    padding: 2rem;
    margin: 1rem 0;
    border-left: 4px solid #667eea;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(102, 126, 234, 0.2);
    color: #e0e0e0;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateX(10px);
    border-left: 4px solid #764ba2;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(118, 75, 162, 0.3);
}

/* Digital Sensor Styling */
.digital-sensor {
    background: rgba(30, 30, 60, 0.9);
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(102, 126, 234, 0.3);
}

.digital-sensor.active {
    background: linear-gradient(135deg, rgba(0, 184, 148, 0.2), rgba(0, 206, 201, 0.2));
    border: 1px solid rgba(0, 184, 148, 0.5);
}

/* Section Headers */
.section-header {
    color: #e0e0e0 !important;
    font-weight: 600 !important;
    font-size: 1.4rem !important;
    margin-bottom: 1.5rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.8rem !important;
    text-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
}

/* APU Status Indicator */
.apu-status {
    display: inline-flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.8rem 1.5rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 1rem;
    margin-left: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.apu-active {
    background: rgba(0, 184, 148, 0.2);
    color: #00b894;
    border: 2px solid rgba(0, 184, 148, 0.5);
    box-shadow: 0 0 20px rgba(0, 184, 148, 0.3);
}

.apu-standby {
    background: rgba(253, 203, 110, 0.2);
    color: #fdcb6e;
    border: 2px solid rgba(253, 203, 110, 0.5);
    box-shadow: 0 0 20px rgba(253, 203, 110, 0.3);
}

.apu-offline {
    background: rgba(225, 112, 85, 0.2);
    color: #e17055;
    border: 2px solid rgba(225, 112, 85, 0.5);
    box-shadow: 0 0 20px rgba(225, 112, 85, 0.3);
}

/* Responsive */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2.2rem;
    }
    
    .block-container {
        padding: 1rem 0.5rem;
    }
    
    .sensor-panel, .results-panel {
        padding: 1.5rem;
    }
    
    .sensor-grid {
        grid-template-columns: 1fr;
    }
    
    .legend-grid {
        grid-template-columns: 1fr;
    }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(20, 20, 40, 0.5);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2, #667eea);
}
</style>
"""

st.markdown(dark_premium_css, unsafe_allow_html=True)

# =========================
# Header Section
# =========================
st.markdown("""
<div class="main-header" style="margin-top: 2rem;">
    <h1>🚇 Metro Train APU Health Monitor</h1>
    <div class="subtitle">Advanced AI-Powered Compressor Diagnostics & Predictive Maintenance System</div>
</div>
""", unsafe_allow_html=True)

# =========================
# Main Layout: Sensors (Left) + Results (Right)
# =========================
left_col, right_col = st.columns([1.1, 0.9], gap="large")

# Initialize session state for prediction results
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction_result = None

with left_col:
    # =========================
    # APU Status and Image Section
    # =========================
    
    # APU Status indicator (simulated based on sensor values)
    col_img, col_status = st.columns([1, 1.2])
    
    with col_img:
        if os.path.exists("machine.jpg"):
            st.image("machine.jpg", width=200)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #30306080, #40407080); height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 15px; color: #a0a0b0; border: 2px dashed rgba(102, 126, 234, 0.3);">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">🚇</div>
                    <div><strong>APU Unit</strong></div>
                    <small style="opacity: 0.7;"></small>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_status:
        # Calculate overall system health from sensor values (simplified)
        if 'sensor_sum' not in st.session_state:
            st.session_state.sensor_sum = 0
        
        # More sophisticated status logic
        if st.session_state.sensor_sum > 50:
            apu_status = "ACTIVE"
            status_class = "apu-active"
            status_emoji = "🟢"
        elif st.session_state.sensor_sum > 20:
            apu_status = "STANDBY"
            status_class = "apu-standby"
            status_emoji = "🟡"
        else:
            apu_status = "OFFLINE"
            status_class = "apu-offline"
            status_emoji = "🔴"
        
        st.markdown(f"""
        <div class="section-header">
            🔧 APU Status 
            <span class="apu-status {status_class}">
                {status_emoji} {apu_status}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # System metrics
        st.markdown(f"""
        <div style="color: #a0a0b0; margin-top: 1rem;">
            <div style="margin: 0.5rem 0;"><strong>System Activity:</strong> {st.session_state.sensor_sum:.1f}/100</div>
            <div style="margin: 0.5rem 0;"><strong>Health Score:</strong> {(st.session_state.sensor_sum/100*100):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Compact Legend
        with st.expander("📋 Sensor Reference Guide", expanded=False):
            sensors_info = [
                ("TP2", "Compressor pressure (bar)"),
                ("TP3", "Pneumatic panel pressure (bar)"),
                ("H1", "Separator pressure drop (bar)"),
                ("DV_pressure", "Towers discharge drop (bar)"),
                ("Reservoirs", "Downstream pressure (bar)"),
                ("Oil_temperature", "Oil temperature (°C)"),
                ("Motor_current", "Motor phase current (A)"),
                ("COMP", "Air intake valve"),
                ("DV_eletric", "Compressor outlet valve"),
                ("Towers", "Tower operation selector"),
                ("MPG", "Compressor start signal"),
                ("LPS", "Low pressure sensor"),
                ("Pressure_switch", "Towers discharge detector"),
                ("Oil_level", "Oil level detector"),
                ("Caudal_impulses", "Air flow pulse counter")
            ]
            
            st.markdown('<div class="legend-grid">', unsafe_allow_html=True)
            for sensor, description in sensors_info:
                sensor_type = "🔢" if sensor in ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"] else "📊"
                st.markdown(f'<div class="legend-item">{sensor_type} <strong>{sensor}:</strong> {description}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Input Method Toggle
    # =========================
    st.markdown('<div class="section-header">⚡ Sensor Configuration Panel</div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "🎛️ Input Method:",
        ["🎚️ Sliders/Switches", "⌨️ Number Input"],
        horizontal=True,
        key="input_method"
    )

    # =========================
    # Sensor Input Panel - Better Grouped
    # =========================
    
    # Define all sensor configurations
    analog_sensors = [
        ("TP2", "📊 TP2 (bar)", -0.1, 10.0, 0.0),
        ("TP3", "📊 TP3 (bar)", 0.0, 15.0, 9.0),
        ("H1", "📊 H1 (bar)", -0.1, 10.0, 0.0),
        ("DV_pressure", "📊 DV Pressure (bar)", -0.1, 3.0, 0.0),
        ("Reservoirs", "📊 Reservoirs (bar)", 0.0, 15.0, 9.0),
        ("Oil_temperature", "📊 Oil Temperature (°C)", 20.0, 80.0, 53.0),
        ("Motor_current", "📊 Motor Current (A)", 0.0, 10.0, 4.0)
    ]
    
    digital_sensors = [
        ("COMP", "🔢 COMP", "Air intake valve active"),
        ("DV_eletric", "🔢 DV Electric", "Outlet valve active"),
        ("Towers", "🔢 Towers", "Tower 2 in operation"),
        ("MPG", "🔢 MPG", "Compressor start signal"),
        ("LPS", "🔢 LPS", "Low pressure detected"),
        ("Pressure_switch", "🔢 Pressure Switch", "Tower discharge detected"),
        ("Oil_level", "🔢 Oil Level", "Low oil level detected"),
        ("Caudal_impulses", "🔢 Caudal Impulses", "Air flow detected")
    ]
    
    sensor_values = {}
    
    # Analog Sensors Section
    st.markdown("""
    <div class="sensor-section">
        <div class="section-header" style="font-size: 1.2rem; margin-bottom: 1rem;">📊 Analog Sensors</div>
    """, unsafe_allow_html=True)
    
    # Create grid for analog sensors
    analog_col1, analog_col2 = st.columns(2)
    for i, (sensor_key, label, min_val, max_val, default) in enumerate(analog_sensors):
        col = analog_col1 if i % 2 == 0 else analog_col2
        with col:
            if input_method == "🎚️ Sliders/Switches":
                sensor_values[sensor_key] = st.slider(
                    label, 
                    min_val, 
                    max_val, 
                    default, 
                    0.01 if max_val <= 1 else 0.1, 
                    key=f"{sensor_key}_slider"
                )
            else:
                sensor_values[sensor_key] = st.number_input(
                    label, 
                    min_val, 
                    max_val, 
                    default, 
                    0.01 if max_val <= 1 else 0.1, 
                    key=f"{sensor_key}_input"
                )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Digital Sensors Section
    st.markdown("""
    <div class="sensor-section">
        <div class="section-header" style="font-size: 1.2rem; margin-bottom: 1rem;">🔢 Digital Sensors</div>
    """, unsafe_allow_html=True)
    
    # Create grid for digital sensors
    digital_col1, digital_col2 = st.columns(2)
    for i, (sensor_key, label, description) in enumerate(digital_sensors):
        col = digital_col1 if i % 2 == 0 else digital_col2
        with col:
            if input_method == "🎚️ Sliders/Switches":
                sensor_values[sensor_key] = 1 if st.checkbox(
                    f"{label}: {description}", 
                    key=f"{sensor_key}_switch"
                ) else 0
            else:
                sensor_values[sensor_key] = st.selectbox(
                    f"{label}: {description}",
                    options=[0, 1],
                    index=0,
                    key=f"{sensor_key}_select"
                )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Update sensor sum for APU status (only analog sensors for meaningful sum)
    analog_sum = sum([
        sensor_values.get('TP2', 0),
        sensor_values.get('TP3', 0),
        sensor_values.get('H1', 0),
        sensor_values.get('Reservoirs', 0),
        sensor_values.get('Oil_temperature', 0),
        sensor_values.get('Motor_current', 0)
    ])
    digital_sum = sum([
        sensor_values.get('COMP', 0),
        sensor_values.get('DV_eletric', 0),
        sensor_values.get('Towers', 0),
        sensor_values.get('MPG', 0),
        sensor_values.get('LPS', 0),
        sensor_values.get('Pressure_switch', 0),
        sensor_values.get('Oil_level', 0),
        sensor_values.get('Caudal_impulses', 0)
    ])
    
    st.session_state.sensor_sum = analog_sum + digital_sum * 5  # Weight digital sensors
    
    # Diagnostic Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Run Advanced Diagnostic Analysis", use_container_width=True, type="primary"):
        # Create input dataframe with correct column order matching your dataset
        input_df = pd.DataFrame([{
            'TP2': sensor_values.get('TP2', 0),
            'TP3': sensor_values.get('TP3', 0), 
            'H1': sensor_values.get('H1', 0),
            'DV_pressure': sensor_values.get('DV_pressure', 0),
            'Reservoirs': sensor_values.get('Reservoirs', 0),
            'Oil_temperature': sensor_values.get('Oil_temperature', 0),
            'Motor_current': sensor_values.get('Motor_current', 0),
            'COMP': sensor_values.get('COMP', 0),
            'DV_eletric': sensor_values.get('DV_eletric', 0),
            'Towers': sensor_values.get('Towers', 0),
            'MPG': sensor_values.get('MPG', 0),
            'LPS': sensor_values.get('LPS', 0),
            'Pressure_switch': sensor_values.get('Pressure_switch', 0),
            'Oil_level': sensor_values.get('Oil_level', 0),
            'Caudal_impulses': sensor_values.get('Caudal_impulses', 0)
        }])

        with st.spinner("🔬 Analyzing sensor data and running AI diagnostics..."):
            try:
                prediction, probability, explanations = predict_with_explanation(input_df)
                explanation_text, shap_image_path = explanations[0]
                pred_label = 'FAILURE' if prediction[0] == 1 else 'NORMAL'
                
                st.session_state.prediction_made = True
                st.session_state.prediction_result = {
                    'pred_label': pred_label,
                    'probability': probability,
                    'explanation_text': explanation_text,
                    'shap_image_path': shap_image_path,
                    'input_data': input_df,
                    'sensor_values': sensor_values
                }
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")


def format_ai_explanation(explanation_text):
    """Format AI explanation with better styling and structure"""
    lines = explanation_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Format section headers
        if line.startswith('DETAILED SENSOR ANALYSIS'):
            formatted_lines.append('<div style="color: #ff6b35; font-weight: 700; font-size: 1.1rem; margin: 1.5rem 0 1rem 0;">🔍 DETAILED SENSOR ANALYSIS</div>')
        elif line.startswith('CRITICAL ISSUES IDENTIFIED'):
            formatted_lines.append('<div style="color: #e74c3c; font-weight: 700; font-size: 1.1rem; margin: 1.5rem 0 1rem 0;">🚨 CRITICAL ISSUES IDENTIFIED</div>')
        elif line.startswith('PRIMARY FACTOR'):
            formatted_lines.append('<div style="color: #fdcb6e; font-weight: 700; font-size: 1.1rem; margin: 1.5rem 0 1rem 0;">📈 PRIMARY FACTOR</div>')
        elif line.startswith('MAINTENANCE RECOMMENDATIONS'):
            formatted_lines.append('<div style="color: #27ae60; font-weight: 700; font-size: 1.1rem; margin: 1.5rem 0 1rem 0;">🛠️ MAINTENANCE RECOMMENDATIONS</div>')
        elif line.startswith('SYSTEM STATUS'):
            formatted_lines.append('<div style="color: #00b894; font-weight: 700; font-size: 1.1rem; margin: 1.5rem 0 1rem 0;">✅ SYSTEM STATUS</div>')
        elif line.startswith('ATTENTION REQUIRED'):
            formatted_lines.append(f'<div style="background: rgba(253, 203, 110, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #fdcb6e; margin: 1rem 0; color: #fff3cd;">⚠️ {line}</div>')
        elif line.startswith('CRITICAL ALERT'):
            formatted_lines.append(f'<div style="background: rgba(231, 76, 60, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #e74c3c; margin: 1rem 0; color: #ffe8e8;">🚨 {line}</div>')
        
        # Format numbered sensor items
        elif line and line[0].isdigit() and '. ' in line:
            formatted_lines.append(f'<div class="sensor-item"><strong style="color: #667eea;">{line}</strong></div>')
        
        # Format impact lines
        elif line.startswith('Impact:'):
            formatted_lines.append(f'<div class="impact-text" style="margin: 1.5rem 0 1rem 0;">📊 {line}</div>')
        
        # Format recommendations
        elif line.startswith('URGENT:') or line.startswith('CRITICAL:') or line.startswith('IMMEDIATE ACTION:') or line.startswith('SAFETY ALERT:'):
            formatted_lines.append(f'<div class="critical-issue" style="margin: 1.5rem 0 1rem 0;">🚨 {line}</div>')
        elif line.startswith('MAINTAIN:') or line.startswith('OPTIMAL:') or line.startswith('MONITOR:'):
            formatted_lines.append(f'<div class="recommendation" style="margin: 1.5rem 0 1rem 0;">✅ {line}</div>')
        
        # Regular text
        else:
            if line:
                formatted_lines.append(f'<div style="margin: 0.5rem 0; line-height: 1.6; color: #e8e8f0;">{line}</div>')
    
    return ''.join(formatted_lines)

# =========================
# Results Panel (Right Side)
# =========================
with right_col:
    st.markdown('<div class="section-header">📊 Diagnostic Results & Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.prediction_made:
        # No prediction yet
        st.markdown("""
        <div class="status-waiting">
            <h3>⏳ No Prediction Yet</h3>
            <p style="margin: 0; opacity: 0.9;">Configure sensors and run diagnostic analysis to view results</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show current sensor summary
        st.markdown('<div class="section-header">📋 Current Sensor Summary</div>', unsafe_allow_html=True)
        
        # Create summary metrics
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">📊 Analog Sensors</div>
                <div style="opacity: 0.8;">TP2: {sensor_values.get('TP2', 0):.3f} bar</div>
                <div style="opacity: 0.8;">TP3: {sensor_values.get('TP3', 0):.3f} bar</div>
                <div style="opacity: 0.8;">H1: {sensor_values.get('H1', 0):.3f} bar</div>
                <div style="opacity: 0.8;">DV Pressure: {sensor_values.get('DV_pressure', 0):.3f} bar</div>
                <div style="opacity: 0.8;">Oil Temp: {sensor_values.get('Oil_temperature', 0):.1f}°C</div>
                <div style="opacity: 0.8;">Motor Current: {sensor_values.get('Motor_current', 0):.2f}A</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_metric2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">🔢 Digital Sensors Status</div>
                <div style="opacity: 0.8;">COMP: {'🟢 Active' if sensor_values.get('COMP', 0) else '🔴 Inactive'}</div>
                <div style="opacity: 0.8;">DV Electric: {'🟢 Active' if sensor_values.get('DV_eletric', 0) else '🔴 Inactive'}</div>
                <div style="opacity: 0.8;">Towers: {'🟢 Tower 2' if sensor_values.get('Towers', 0) else '🔴 Tower 1'}</div>
                <div style="opacity: 0.8;">MPG: {'🟢 Active' if sensor_values.get('MPG', 0) else '🔴 Inactive'}</div>
                <div style="opacity: 0.8;">LPS: {'🟡 Low Pressure' if sensor_values.get('LPS', 0) else '🟢 Normal'}</div>
                <div style="opacity: 0.8;">Oil Level: {'🔴 Low' if sensor_values.get('Oil_level', 0) else '🟢 Normal'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">⚙️ System Health Overview</div>
            <div style="opacity: 0.8;">Analog Activity Score: {analog_sum:.1f}</div>
            <div style="opacity: 0.8;">Digital Systems Active: {digital_sum}/8</div>
            <div style="opacity: 0.8;">Overall Health: {(st.session_state.sensor_sum/100*100):.1f}%</div>
            <div style="opacity: 0.8;">APU Status: <strong>{apu_status}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Show prediction results
        result = st.session_state.prediction_result
        
        # Status display
        if result['pred_label'] == 'NORMAL':
            st.markdown(f"""
            <div class="status-normal">
                <h3>✅ SYSTEM STATUS: NORMAL OPERATION</h3>
                <p style="margin: 0; font-size: 1.3rem; opacity: 0.9;">Confidence: {float(result['probability'][0]):.1%}</p>
                <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">All systems operating within normal parameters</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-failure">
                <h3>⚠️ SYSTEM STATUS: FAILURE RISK DETECTED</h3>
                <p style="margin: 0; font-size: 1.3rem; opacity: 0.9;">Confidence: {float(result['probability'][0]):.1%}</p>
                <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">Immediate attention required - potential system failure predicted</div>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced AI Explanation with proper formatting
        st.markdown('<div class="section-header">🧠 AI Analysis Explanation</div>', unsafe_allow_html=True)
        
        explanation_text = result['explanation_text']
        explanation_text = explanation_text.split(":", 1)[1].strip() if explanation_text.startswith("Row 0:") else explanation_text
        
        # Format the explanation with enhanced styling
        formatted_explanation = format_ai_explanation(explanation_text)
        st.markdown(f"""
        <div class="ai-explanation">
            <div class="explanation-header">🤖 Detailed AI Analysis Report</div>
            <div class="explanation-text">{formatted_explanation}</div>
        </div>
        """, unsafe_allow_html=True)

        # Current Sensor Values Summary
        st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
        
        col_anal1, col_anal2 = st.columns(2)
        
      
        # SHAP Visualization
        st.markdown('<div class="section-header">📈 Feature Impact Analysis (SHAP)</div>', unsafe_allow_html=True)
        if os.path.exists(result['shap_image_path']):
            st.image(result['shap_image_path'], caption="SHAP Analysis: Most Influential Factors in Prediction", use_container_width=True)
        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                <div style="color: #fdcb6e; font-weight: 600;">SHAP Visualization Unavailable</div>
                <div style="color: #a0a0b0; margin-top: 0.5rem; font-size: 0.9rem;">Ensure the SHAP image path is correct</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons including download
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔄 Reset Analysis", use_container_width=True):
                st.session_state.prediction_made = False
                st.session_state.prediction_result = None
                st.rerun()
        
        with col_btn2:
            if st.button("📊 Re-run", use_container_width=True):
                st.session_state.prediction_made = False
                st.rerun()
        
        with col_btn3:
            # Download report button
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_content = generate_detailed_report(result, result['sensor_values'], timestamp)
            filename = f"APU_Health_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name=filename,
                mime="text/plain",
                use_container_width=True,
                key="download_report"
            )

# =========================
# Footer
# =========================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: rgba(15, 15, 35, 0.95); border-radius: 25px; margin-top: 2rem; backdrop-filter: blur(20px); border: 1px solid rgba(102, 126, 234, 0.2); box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);">
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
        Metro Train APU Health Monitor v2.0 🚀
    </div>
    <div style="color: #a0a0b0; font-size: 0.9rem;">
        Powered by Explainable AI & Advanced Machine Learning<br>
        <span style="color: #667eea;">Copyright © 2025 Odubiyi Ifeoluwa Antonia</span>
    </div>
</div>
""", unsafe_allow_html=True)