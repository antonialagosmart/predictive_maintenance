import streamlit as st 
import pandas as pd
import os
from explain_predictor import predict_with_explanation

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
    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
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

/* Image Container */
.image-container {
    background: rgba(30, 30, 60, 0.9);
    border-radius: 25px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(102, 126, 234, 0.2);
    border: 1px solid rgba(102, 126, 234, 0.2);
}

/* Legend Styling */
.legend-item {
    background: rgba(30, 30, 60, 0.7);
    border-radius: 10px;
    padding: 0.8rem;
    margin: 0.5rem 0;
    border-left: 3px solid #667eea;
    font-size: 0.9rem;
    color: #c0c0d0;
    transition: all 0.3s ease;
}

.legend-item:hover {
    background: rgba(40, 40, 70, 0.9);
    border-left: 3px solid #764ba2;
    transform: translateX(5px);
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

/* Glowing effects */
.glow {
   
}

@keyframes glow {
    from {
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
    }
    to {
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.8);
    }
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
  
        if os.path.exists("maBchine.jpg"):
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
        if st.session_state.sensor_sum > 300:
            apu_status = "ACTIVE"
            status_class = "apu-active"
            status_emoji = "🟢"
        elif st.session_state.sensor_sum > 100:
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
            <div style="margin: 0.5rem 0;"><strong>System Activity:</strong> {st.session_state.sensor_sum:.1f}/1500</div>
            <div style="margin: 0.5rem 0;"><strong>Health Score:</strong> {(st.session_state.sensor_sum/1500*100):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Legend
        with st.expander("📋 Sensor Reference Guide", expanded=False):
            sensors_info = [
                ("DV_pressure", "Differential Valve Pressure"),
                ("DV_eletric", "Differential Valve Electric Status"),
                ("H1", "Cyclonic Separator Pressure Drop"),
                ("COMP", "Compressor Status"),
                ("MPG", "Main Pressure Gauge"),
                ("TP2 / TP3", "Compressor Stage Pressure Sensors"),
                ("Oil_temperature", "Compressor Oil Temperature"),
                ("Towers", "Dryer/Tower Status"),
                ("LPS", "Low-Pressure Sensor"),
                ("Reservoirs", "Storage Reservoirs Pressure"),
                ("Motor_current", "Compressor Motor Current"),
                ("Pressure_switch", "ON/OFF State of Compressor"),
                ("Oil_level", "Compressor Oil Level"),
                ("Caudal_impulses", "Flow Pulses Representing Airflow")
            ]
            
            for sensor, description in sensors_info:
                st.markdown(f'<div class="legend-item"><strong>{sensor}:</strong> {description}</div>', unsafe_allow_html=True)

    # =========================
    # Input Method Toggle
    # =========================
    st.markdown('<div class="section-header">⚡ Sensor Configuration Panel</div>', unsafe_allow_html=True)
    
    
    input_method = st.radio(
        "🎛️ Input Method:",
        ["🎚️ Sliders", "⌨️ Number Input"],
        horizontal=True,
        key="input_method"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Sensor Input Panel
    # =========================
    # Two columns for sensors
    sensor_col1, sensor_col2 = st.columns(2)
    
    # Define sensor configurations
    sensors_config = [
        ("LPS", "🔧 LPS", 25.0),
        ("COMP", "⚙️ COMP", 50.0),
        ("Oil_temperature", "🌡️ Oil Temperature", 30.0),
        ("TP3", "📈 TP3", 40.0),
        ("DV_pressure", "💨 DV Pressure", 35.0),
        ("DV_eletric", "⚡ DV Electric", 45.0),
        ("Pressure_switch", "🔌 Pressure Switch", 60.0)
    ]
    
    sensors_config_col2 = [
        ("MPG", "📊 MPG", 55.0),
        ("Motor_current", "🔋 Motor Current", 40.0),
        ("TP2", "📈 TP2", 45.0),
        ("H1", "📏 H1", 30.0),
        ("Reservoirs", "🛢️ Reservoirs", 50.0),
        ("Towers", "🗼 Towers", 35.0),
        ("Oil_level", "🛢️ Oil Level", 70.0)
    ]
    
    sensor_values = {}
    
    with sensor_col1:
        for sensor_key, label, default in sensors_config:
            if input_method == "🎚️ Sliders":
                sensor_values[sensor_key] = st.slider(label, 0.0, 100.0, default, 0.1, key=f"{sensor_key}_slider")
            else:
                sensor_values[sensor_key] = st.number_input(label, 0.0, 100.0, default, 0.1, key=f"{sensor_key}_input")
        
    with sensor_col2:
        for sensor_key, label, default in sensors_config_col2:
            if input_method == "🎚️ Sliders":
                sensor_values[sensor_key] = st.slider(label, 0.0, 100.0, default, 0.1, key=f"{sensor_key}_slider")
            else:
                sensor_values[sensor_key] = st.number_input(label, 0.0, 100.0, default, 0.1, key=f"{sensor_key}_input")
    
    # Update sensor sum for APU status
    st.session_state.sensor_sum = sum(sensor_values.values())
    
    # Additional sensor
    if input_method == "🎚️ Sliders":
        sensor_values["Caudal_impulses"] = st.slider("🌀 Caudal Impulses", 0.0, 100.0, 25.0, 0.1, key="Caudal_impulses_slider")
    else:
        sensor_values["Caudal_impulses"] = st.number_input("🌀 Caudal Impulses", 0.0, 100.0, 25.0, 0.1, key="Caudal_impulses_input")
    
    # Diagnostic Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Run Advanced Diagnostic Analysis", use_container_width=True, type="primary"):
        input_df = pd.DataFrame([{
            'LPS': sensor_values['LPS'], 'MPG': sensor_values['MPG'], 'COMP': sensor_values['COMP'], 
            'Motor_current': sensor_values['Motor_current'], 'Oil_temperature': sensor_values['Oil_temperature'], 
            'TP2': sensor_values['TP2'], 'TP3': sensor_values['TP3'], 'H1': sensor_values['H1'],
            'DV_pressure': sensor_values['DV_pressure'], 'Reservoirs': sensor_values['Reservoirs'], 
            'DV_eletric': sensor_values['DV_eletric'], 'Towers': sensor_values['Towers'], 
            'Pressure_switch': sensor_values['Pressure_switch'], 'Oil_level': sensor_values['Oil_level'],
            'Caudal_impulses': sensor_values['Caudal_impulses']
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
    
    st.markdown('</div>', unsafe_allow_html=True)

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
                <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">🔧 Primary Systems</div>
                <div style="opacity: 0.8;">LPS: {sensor_values.get('LPS', 0):.1f}</div>
                <div style="opacity: 0.8;">COMP: {sensor_values.get('COMP', 0):.1f}</div>
                <div style="opacity: 0.8;">MPG: {sensor_values.get('MPG', 0):.1f}</div>
                <div style="opacity: 0.8;">Motor: {sensor_values.get('Motor_current', 0):.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_metric2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">🌡️ Temperature & Pressure</div>
                <div style="opacity: 0.8;">Oil Temp: {sensor_values.get('Oil_temperature', 0):.1f}°</div>
                <div style="opacity: 0.8;">TP2: {sensor_values.get('TP2', 0):.1f}</div>
                <div style="opacity: 0.8;">TP3: {sensor_values.get('TP3', 0):.1f}</div>
                <div style="opacity: 0.8;">DV Pressure: {sensor_values.get('DV_pressure', 0):.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">⚙️ System Health Overview</div>
            <div style="opacity: 0.8;">Total Sensor Activity: {st.session_state.sensor_sum:.1f}/1500</div>
            <div style="opacity: 0.8;">Health Score: {(st.session_state.sensor_sum/1500*100):.1f}%</div>
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

        # AI Explanation
        st.markdown('<div class="section-header">🧠 AI Analysis Explanation</div>', unsafe_allow_html=True)
        explanation_text = result['explanation_text']
        explanation_text = explanation_text.split(":", 1)[1].strip() if explanation_text.startswith("Row 0:") else explanation_text
        
       
        for line in explanation_text.split("\n"):
            if line.strip():
                st.markdown(f"<div style='margin: 0.8rem 0; padding: 0.5rem; background: rgba(102, 126, 234, 0.1); border-left: 3px solid #667eea; border-radius: 5px;'>• {line.strip()}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Key Sensor Values Summary
        st.markdown('<div class="section-header">📋 Key Sensor Values at Analysis</div>', unsafe_allow_html=True)
        
        col_sens1, col_sens2 = st.columns(2)
        
        with col_sens1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #667eea;">🔧 Critical Systems</div>
                <div style="margin: 0.3rem 0;">LPS: <strong>{result['sensor_values']['LPS']:.1f}</strong></div>
                <div style="margin: 0.3rem 0;">COMP: <strong>{result['sensor_values']['COMP']:.1f}</strong></div>
                <div style="margin: 0.3rem 0;">Oil Level: <strong>{result['sensor_values']['Oil_level']:.1f}</strong></div>
                <div style="margin: 0.3rem 0;">Motor Current: <strong>{result['sensor_values']['Motor_current']:.1f}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_sens2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #764ba2;">🌡️ Monitoring Points</div>
                <div style="margin: 0.3rem 0;">Oil Temp: <strong>{result['sensor_values']['Oil_temperature']:.1f}°</strong></div>
                <div style="margin: 0.3rem 0;">TP2: <strong>{result['sensor_values']['TP2']:.1f}</strong></div>
                <div style="margin: 0.3rem 0;">TP3: <strong>{result['sensor_values']['TP3']:.1f}</strong></div>
                <div style="margin: 0.3rem 0;">DV Pressure: <strong>{result['sensor_values']['DV_pressure']:.1f}</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP Visualization
        st.markdown('<div class="section-header">📈 Feature Impact Analysis (SHAP)</div>', unsafe_allow_html=True)
        if os.path.exists(result['shap_image_path']):
          
            st.image(result['shap_image_path'], caption="SHAP Analysis: Most Influential Factors in Prediction", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                <div style="color: #fdcb6e; font-weight: 600;">SHAP Visualization Unavailable</div>
                <div style="color: #a0a0b0; margin-top: 0.5rem; font-size: 0.9rem;">Ensure the SHAP image path is correct</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔄 Reset Analysis", use_container_width=True):
                st.session_state.prediction_made = False
                st.session_state.prediction_result = None
                st.rerun()
        
        with col_btn2:
            if st.button("📊 Re-run Analysis", use_container_width=True):
                st.session_state.prediction_made = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

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