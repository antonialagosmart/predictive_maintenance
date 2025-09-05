import streamlit as st 
import pandas as pd
import os
from explain_predictor import predict_with_explanation

# Page config
st.set_page_config(page_title="Machine Health Prediction", layout="wide")

st.title("🛠️ Explainable Predictive Maintenance for Smart Metro Train Compressor's Air Production Unit (APU)")

# Image and Feature Legend side by side
col_img, col_legend = st.columns([1, 2])

with col_img:
    if os.path.exists("machine.jpg"):
        st.image("machine.jpg", width=200)
    else:
        st.warning("⚠️ 'machine.jpg' not found. Please add the image to the project directory.")

with col_legend:
    with st.expander("◽ Feature Legend: Key Sensor Definitions", expanded=False):
        st.markdown("""
        The following features were identified as **most influential** in the model's predictions:

        - **DV_pressure (Differential Valve Pressure)** – Measures the pressure difference across the delivery valve.  
        - **DV_eletric (Differential Valve Electric Status)** – Electrical signal status controlling the delivery valve.  
        - **H1 (Cyclonic Separator Pressure Drop)** – Indicates moisture/contaminant resistance in the air stream.  
        - **COMP (Compressor Status)** – Operational load/health of the compressor.  
        - **MPG (Main Pressure Gauge)** – Represents system-wide main air pressure levels.  
        - **TP2 (Compressor Pressure Sensor 2)** – Tracks secondary stage compressor pressure.  
        - **TP3 (Compressor Pressure Sensor 3)** – Monitors tertiary stage compressor pressure.  
        - **Oil_temperature (Compressor Oil Temperature)** – Ensures lubrication stability and overheating detection.  
        - **Towers (Dryer/Tower Status)** – Reflects tower dryer load or regeneration status.  

        Other available features:  
        - **LPS (Low-Pressure Sensor)** – Air pressure in low-pressure zone.  
        - **Reservoirs** – Storage reservoir air levels or pressure.  
        - **Motor_current** – Current drawn by the compressor motor.  
        - **Pressure_switch** – Indicates ON/OFF state of compressor pressure control.  
        - **Oil_level** – Compressor oil level, ensuring lubrication is sufficient.  
        - **Caudal_impulses** – Flow pulses representing airflow rate.  
        """)

# Input sliders
st.subheader("◽ Enter Machine Sensor Readings")
padding_bottom = '<div style="padding-bottom: 1.5rem;"></div>'

def padded_slider_input(label, key_prefix, default=0.0):
    col, _ = st.columns([0.95, 0.05])
    with col:
        slider_val = st.slider(label, 0.0, 100.0, default, 0.1, key=f"{key_prefix}_slider")
        number_val = st.number_input(
            label="",
            min_value=0.0,
            max_value=100.0,
            value=slider_val,
            step=0.1,
            key=f"{key_prefix}_input",
            label_visibility="collapsed"
        )
        st.markdown(padding_bottom, unsafe_allow_html=True)
        return number_val if abs(slider_val - number_val) > 0.0001 else slider_val

with st.expander("⚙️ Show Sliders Panel", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        LPS = padded_slider_input("🔧 LPS", "LPS")
        COMP = padded_slider_input("⚙️ COMP", "COMP")
        Oil_temperature = padded_slider_input("🌡️ Oil Temperature", "Oil_temperature")
        TP3 = padded_slider_input("📈 TP3", "TP3")
        DV_pressure = padded_slider_input("💨 DV Pressure", "DV_pressure")
        DV_eletric = padded_slider_input("⚡ DV Electric", "DV_eletric")
        Pressure_switch = padded_slider_input("🔌 Pressure Switch", "Pressure_switch")
        Caudal_impulses = padded_slider_input("🌀 Caudal Impulses", "Caudal_impulses")
    with col2:
        MPG = padded_slider_input("📄 MPG", "MPG")                            
        Motor_current = padded_slider_input("🔋 Motor Current", "Motor_current")  
        TP2 = padded_slider_input("📈 TP2", "TP2")                            
        H1 = padded_slider_input("📏 H1", "H1")                               
        Reservoirs = padded_slider_input("🛢️ Reservoirs", "Reservoirs")     
        Towers = padded_slider_input("🗼 Towers", "Towers")                  
        Oil_level = padded_slider_input("🔧 Oil Level", "Oil_level")       

# Prediction button
if st.button("⚙️ Start Diagnostic Scan"):
    input_df = pd.DataFrame([{
        'LPS': LPS, 'MPG': MPG, 'COMP': COMP, 'Motor_current': Motor_current,
        'Oil_temperature': Oil_temperature, 'TP2': TP2, 'TP3': TP3, 'H1': H1,
        'DV_pressure': DV_pressure, 'Reservoirs': Reservoirs, 'DV_eletric': DV_eletric,
        'Towers': Towers, 'Pressure_switch': Pressure_switch, 'Oil_level': Oil_level,
        'Caudal_impulses': Caudal_impulses
    }])

    st.write("⏳ Running prediction...")

    prediction, probability, explanations = predict_with_explanation(input_df)
    explanation_text, shap_image_path = explanations[0]
    pred_label = 'FAILURE' if prediction[0] == 1 else 'NORMAL'
    st.success(f"Prediction: {pred_label} (Probability: {float(probability[0]):.2f})")

    # Show explanation
    explanation_text = explanation_text.split(":", 1)[1].strip() if explanation_text.startswith("Row 0:") else explanation_text
    st.markdown("### 🧠 Explanation of Prediction")
    for line in explanation_text.split("\n"):
        if line.strip():
            st.markdown(line.strip())

    st.markdown("### ◽ Feature Contribution (SHAP)")
    if os.path.exists(shap_image_path):
        st.image(shap_image_path, caption="Top 5 SHAP Features", use_container_width=True)
    else:
        st.warning("⚠️ SHAP image not found.")

st.markdown("---")
st.markdown("Copyright © 2025 Odubiyi Ifeoluwa Antonia")
