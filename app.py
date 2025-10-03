import streamlit as st 
import pandas as pd
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from explain_predictor import predict_with_explanation, generate_detailed_report
from data_explainability import SensorValidator
from operational_monitor import OperationalMonitor
from storage_manager import StorageManager
from PIL import Image
import json
from fpdf import FPDF
import base64
from io import BytesIO
import matplotlib.pyplot as plt


def format_explanation_text(explanation_text: str) -> None:
    """Format and display explanation text using Streamlit components"""
    sections = explanation_text.split('\n\n')
    
    for section in sections:
        if not section.strip():
            continue
            
        if "The AI model predicts" in section:
            pred_type = "NORMAL" if "NORMAL" in section else "FAILURE"
            confidence = section.split("confidence of")[1].strip().rstrip(".")
            
            # Use st.container for prediction header
            with st.container():
                color = "rgb(16, 185, 129)" if pred_type == "NORMAL" else "rgb(239, 68, 68)"
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(145deg, {color}22, {color}44);
                             border: 2px solid {color};
                             border-radius: 15px;
                             padding: 20px;
                             margin-bottom: 20px;'>
                        <h2 style='color: {color}; margin: 0; font-size: 24px;'>
                            🤖 AI Prediction: {pred_type}
                        </h2>
                        <p style='color: #94a3b8; margin: 10px 0 0 0;'>
                            Model Confidence: {confidence}
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
        elif "DETAILED SENSOR ANALYSIS" in section:
            st.subheader("🔍 DETAILED SENSOR ANALYSIS")
            
        elif section.startswith(("1.", "2.", "3.", "4.", "5.")):
            # Parse sensor reading and impact
            if "Impact:" in section:
                reading, impact = section.split("Impact:")
                sensor_num = reading.split(".")[0]
                
                # Create expandable section for each sensor
                with st.expander(f"Sensor {sensor_num} Analysis", expanded=True):
                    st.markdown(reading.split(".", 1)[1].strip())
                    
                    # Color-code impact based on content
                    impact = impact.strip()
                    if "NORMAL" in impact:
                        st.success(f"**Impact:** {impact}")
                    else:
                        st.error(f"**Impact:** {impact}")
            
        elif "PRIMARY FACTOR" in section:
            st.subheader("📈 PRIMARY FACTOR")
            st.info(section.replace("PRIMARY FACTOR", "").strip())
            
        elif "MAINTENANCE RECOMMENDATIONS" in section:
            st.subheader("🛠️ MAINTENANCE RECOMMENDATIONS")
            recommendations = section.replace("MAINTENANCE RECOMMENDATIONS", "").strip().split('\n')
            
            for rec in recommendations:
                if rec.strip():
                    if rec.startswith("✅"):
                        st.success(rec)
                    elif rec.startswith("⚠️"):
                        st.warning(rec)
                    elif rec.startswith("OPTIMAL"):
                        st.info(f"✨ {rec}")
                    elif rec.startswith("MONITOR"):
                        st.warning(f"⚠️ {rec}")
                    else:
                        st.markdown(rec)


# Initialize managers
if 'storage_manager' not in st.session_state:
    st.session_state.storage_manager = StorageManager()
if 'operational_monitor' not in st.session_state:
    st.session_state.operational_monitor = OperationalMonitor()
if 'sensor_validator' not in st.session_state:
    st.session_state.sensor_validator = SensorValidator()
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'machine_status' not in st.session_state:
    st.session_state.machine_status = 'ACTIVE'

# Page config
st.set_page_config(
    page_title="APU Health Monitor Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS (keeping your existing styles)
dark_premium_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #4d4d4d 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* All your existing CSS styles remain the same */
.main-header {
    background: rgba(15, 15, 35, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
}

.section-header {
    color: #e0e0e0;
    font-weight: 600;
    font-size: 1.4rem;
    margin-bottom: 1.5rem;
}

.alert-critical {
    background: rgba(231, 76, 60, 0.2);
    border-left: 4px solid #e74c3c;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: #ffe8e8;
}

.alert-warning {
    background: rgba(253, 203, 110, 0.2);
    border-left: 4px solid #fdcb6e;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: #fff3cd;
}

.alert-normal {
    background: rgba(0, 184, 148, 0.2);
    border-left: 4px solid #00b894;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: #d4f5e9;
}

.metric-card {
    background: rgba(30, 30, 60, 0.9);
    border-radius: 18px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid #667eea;
    color: #e0e0e0;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border-radius: 50px !important;
    padding: 1rem 2rem !important;
    font-weight: 600 !important;
    width: 100% !important;
}
</style>
"""

st.markdown(dark_premium_css, unsafe_allow_html=True)

# Header
# Header
# Header with conditional image or fallback
# Header with image and text side by side
st.markdown("""
<style>
.header-container {
    display: flex;
    align-items: center;
    gap: 20px;
    background: rgba(15, 15, 35, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.header-image {
    flex: 0 0 120px;
    text-align: center;
}
.header-text {
    flex: 1;
}
.header-text h1 {
    margin: 0;
    color: white;
}
.subtitle {
    color: #94a3b8;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Create a container for both image and text
col1, col2 = st.columns([1, 4])

with col1:
    # Image component
    if os.path.exists("machine.jpg"):
        try:
            image = Image.open("machine.jpg")
            st.image(image, width=120, caption="APU Unit")
        except Exception as e:
            print(f"Error loading image: {e}")
            st.markdown("🚇", unsafe_allow_html=True)
    else:
        st.markdown("🚇", unsafe_allow_html=True)

with col2:
    # Header text component
    st.markdown("""
        <div class="header-text">
            <h1>Metro Train APU Monitor</h1>
            <div class="subtitle">Advanced IOT-ML Diagnostics with Real-Time Explainability & Trend Analysis</div>
        </div>
    """, unsafe_allow_html=True)


# ========== SIDEBAR: Role Selection & Navigation ==========
with st.sidebar:
    st.markdown("### 👤 User Role Selection")
    role = st.selectbox(
        "Select your role:",
        ["plant_manager", "maintenance_engineer", "ml_engineer"],
        format_func=lambda x: {
            "plant_manager": "🏭 Plant Manager",
            "maintenance_engineer": "🔧 Maintenance Engineer",
            "ml_engineer": "🤖 ML Engineer"
        }[x]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Select view:",
        ["Live Diagnostics", "Trend Analysis"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Role Information")
    role_info = {
        "plant_manager": "**Plant Manager View**\n\nHigh-level operational insights, business impact assessment, and actionable recommendations for production decisions.",
        "maintenance_engineer": "**Maintenance Engineer View**\n\nDetailed technical diagnostics, sensor-level analysis, and specific maintenance procedures for equipment care.",
        "ml_engineer": "**ML Engineer View**\n\nModel performance metrics, data quality assessment, concept drift detection, and system monitoring insights."
    }
    st.info(role_info[role])

# ========== PAGE 1: LIVE DIAGNOSTICS ==========
if page == "Live Diagnostics":
    # Machine Status Display (Add this at the top of the page)
    machine_status = st.session_state.get('machine_status', 'ACTIVE')
    status_colors = {
        'ACTIVE': '#00b894',
        'OFFLINE': '#e74c3c',
        'MAINTENANCE': '#fdcb6e',
        'STANDBY': '#0984e3'
    }
    st.markdown(f"""
        <div style="position: absolute; top: 10px; right: 20px; padding: 10px 20px; 
                    background: rgba(0,0,0,0.2); border-radius: 50px; border: 2px solid {status_colors[machine_status]}">
            <span style="color: {status_colors[machine_status]}; font-weight: bold;">● {machine_status}</span>
        </div>
    """, unsafe_allow_html=True)

    # Add template download button
    st.markdown("### 📥 Data Templates")
    template_data = {
        'timestamp': ['2024-02-20 09:00:00', '2024-02-20 09:15:00'],
        'TP2': [7.5, 7.8],
        'TP3': [8.0, 8.2],
        'H1': [0.5, 0.6],
        'DV_pressure': [1.0, 1.1],
        'Reservoirs': [8.1, 8.3],
        'Oil_temperature': [55, 56],
        'Motor_current': [4.5, 4.6],
        'COMP': [0, 0],
        'DV_eletric': [1, 1],
        'Towers': [0, 1],
        'MPG': [0, 0],
        'LPS': [0, 0],
        'Pressure_switch': [0, 0],
        'Oil_level': [0, 0],
        'Caudal_impulses': [1, 1]
    }
    template_df = pd.DataFrame(template_data)
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        "📥 Download Template CSV",
        template_csv,
        "apu_readings_template.csv",
        "text/csv",
        key='download-template'
    )

    # Add file upload
    uploaded_file = st.file_uploader("📤 Upload Readings CSV", type=['csv'])
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(input_df.head())
            
            if st.button("🔍 Analyze Uploaded Data"):
                results = []
                progress_bar = st.progress(0)
                
                with st.spinner("Processing readings..."):
                    for idx, row in input_df.iterrows():
                        sensor_values = row.to_dict()
                        if 'timestamp' in sensor_values:
                            timestamp = sensor_values.pop('timestamp')
                        else:
                            timestamp = datetime.now().isoformat()
                        
                        # Get data quality report
                        quality_report = st.session_state.sensor_validator.generate_data_quality_report(
                            sensor_values, role)
                        
                        # Make prediction
                        prediction, probability, explanations = predict_with_explanation(pd.DataFrame([sensor_values]))
                        explanation_text, shap_image_path = explanations[0]
                        
                        # Store results (convert numpy types to Python types)
                        result = {
                            'timestamp': timestamp,
                            'prediction': int(prediction[0]),  # Convert to regular int
                            'probability': float(probability[0]),  # Convert to regular float
                            'sensor_values': sensor_values,
                            'data_quality_score': float(quality_report['data_quality_score']),  # Convert to regular float
                            'quality_factors': quality_report['quality_factors'],
                            'explanation_text': explanation_text
                        }
                        results.append(result)
                        
                        # Save to storage manager
                        st.session_state.storage_manager.save_prediction(result)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(input_df))
                
                # Show summary
                st.success(f"✅ Processed {len(results)} readings")
                
                # Display results summary
                df_results = pd.DataFrame([{
                    'timestamp': r['timestamp'],
                    'prediction': r['prediction'],
                    'probability': r['probability'],
                    'data_quality_score': r['data_quality_score']
                } for r in results])
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    failures = len(df_results[df_results['prediction'] == 1])
                    st.metric("Failures Detected", failures)
                with col2:
                    avg_quality = df_results['data_quality_score'].mean()
                    st.metric("Avg Data Quality", f"{avg_quality:.1f}")
                with col3:
                    avg_prob = df_results['probability'].mean()
                    st.metric("Avg Confidence", f"{avg_prob:.1%}")
                
                # Common data quality factors
                st.markdown("### 📊 Data Quality Analysis")
                quality_counts = {}
                for result in results:
                    for factor in result.get('quality_factors', []):
                        key = f"{factor['sensor']}: {factor['reason']}"
                        quality_counts[key] = quality_counts.get(key, 0) + 1
                
                if quality_counts:
                    st.markdown("#### Most Common Quality Issues:")
                    sorted_issues = sorted(quality_counts.items(), key=lambda x: x[1], reverse=True)
                    for issue, count in sorted_issues[:5]:
                        percentage = (count / len(results)) * 100
                        st.warning(f"🔸 {issue} (Found in {percentage:.1f}% of readings)")
                
                # Show detailed results in expandable section
                with st.expander("View Detailed Results"):
                    st.dataframe(df_results)
                    
                # Download detailed report
                report_data = {
                    'summary': {
                        'total_readings': len(results),
                        'failure_count': failures,
                        'avg_quality_score': avg_quality,
                        'avg_confidence': avg_prob
                    },
                    'quality_issues': quality_counts,
                    'detailed_results': results
                }
                
                report_json = json.dumps(report_data, indent=2)
                st.download_button(
                    "📥 Download Detailed Analysis Report",
                    report_json,
                    "batch_analysis_report.json",
                    "application/json"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    
    with left_col:
        st.markdown('<div class="section-header">⚡ Sensor Configuration Panel</div>', unsafe_allow_html=True)
        
        # Input method toggle
        input_method = st.radio(
            "🎛️ Input Method:",
            ["🎚️ Sliders/Switches", "⌨️ Number Input"],
            horizontal=True
        )
        
        # Sensor inputs (keeping your existing structure)
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
        
        # Analog sensors
        st.markdown('<div class="sensor-section"><div class="section-header" style="font-size: 1.2rem;">📊 Analog Sensors</div>', unsafe_allow_html=True)
        analog_col1, analog_col2 = st.columns(2)
        for i, (sensor_key, label, min_val, max_val, default) in enumerate(analog_sensors):
            col = analog_col1 if i % 2 == 0 else analog_col2
            with col:
                if input_method == "🎚️ Sliders/Switches":
                    sensor_values[sensor_key] = st.slider(label, min_val, max_val, default, 0.01 if max_val <= 1 else 0.1, key=f"{sensor_key}_slider")
                else:
                    sensor_values[sensor_key] = st.number_input(label, min_val, max_val, default, 0.01 if max_val <= 1 else 0.1, key=f"{sensor_key}_input")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Digital sensors
        st.markdown('<div class="sensor-section"><div class="section-header" style="font-size: 1.2rem;">🔢 Digital Sensors</div>', unsafe_allow_html=True)
        digital_col1, digital_col2 = st.columns(2)
        for i, (sensor_key, label, description) in enumerate(digital_sensors):
            col = digital_col1 if i % 2 == 0 else digital_col2
            with col:
                if input_method == "🎚️ Sliders/Switches":
                    sensor_values[sensor_key] = 1 if st.checkbox(f"{label}: {description}", key=f"{sensor_key}_switch") else 0
                else:
                    sensor_values[sensor_key] = st.selectbox(f"{label}: {description}", options=[0, 1], index=0, key=f"{sensor_key}_select")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_diagnostic = st.button("🔍 Run Diagnostic Analysis", use_container_width=True, type="primary")
        with col_btn2:
            save_prediction = st.button("💾 Save Prediction", use_container_width=True, disabled=not st.session_state.get('prediction_made', False))
    
    with right_col:
        # Add Sensor Reference Guide at the top
        with st.expander("📋 Sensor Reference Guide", expanded=False):
            show_ranges = st.toggle("Show Operating Ranges", value=False)
            
            st.markdown("""
            <style>
            .sensor-table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }
            .sensor-table th {
                background: rgba(102, 126, 234, 0.1);
                padding: 8px;
                text-align: left;
                border-bottom: 2px solid rgba(102, 126, 234, 0.2);
            }
            .sensor-table td {
                padding: 6px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .sensor-range {
                color: #94a3b8;
                font-size: 0.9em;
            }
            </style>
            """, unsafe_allow_html=True)

            # Organize sensors by type
            st.markdown("#### Analog Sensors")
            if show_ranges:
                st.markdown("""
                <table class="sensor-table">
                    <tr>
                        <th>Sensor Code</th>
                        <th>Description</th>
                        <th>Normal Range</th>
                        <th>Critical Range</th>
                    </tr>
                    <tr>
                        <td><strong>TP2</strong></td>
                        <td>Compressor pressure</td>
                        <td><span class="sensor-range">6.0 - 10.0 bar</span></td>
                        <td><span class="sensor-range">2.0 - 9.5 bar</span></td>
                    </tr>
                    <tr>
                        <td><strong>TP3</strong></td>
                        <td>Pneumatic panel pressure</td>
                        <td><span class="sensor-range">7.0 - 11.0 bar</span></td>
                        <td><span class="sensor-range">6.0 - 10.0 bar</span></td>
                    </tr>
                    <tr>
                        <td><strong>H1</strong></td>
                        <td>Separator pressure drop</td>
                        <td><span class="sensor-range">0.1 - 2.0 bar</span></td>
                        <td><span class="sensor-range">0.05 - 1.8 bar</span></td>
                    </tr>
                    <tr>
                        <td><strong>DV_pressure</strong></td>
                        <td>Towers discharge drop</td>
                        <td><span class="sensor-range">0.0 - 2.5 bar</span></td>
                        <td><span class="sensor-range">-0.1 - 2.0 bar</span></td>
                    </tr>
                    <tr>
                        <td><strong>Reservoirs</strong></td>
                        <td>Downstream pressure</td>
                        <td><span class="sensor-range">7.0 - 11.0 bar</span></td>
                        <td><span class="sensor-range">6.5 - 10.5 bar</span></td>
                    </tr>
                    <tr>
                        <td><strong>Oil_temperature</strong></td>
                        <td>Oil temperature</td>
                        <td><span class="sensor-range">40.0 - 65.0 °C</span></td>
                        <td><span class="sensor-range">25.0 - 70.0 °C</span></td>
                    </tr>
                    <tr>
                        <td><strong>Motor_current</strong></td>
                        <td>Motor phase current</td>
                        <td><span class="sensor-range">0.0 - 9.0 A</span></td>
                        <td><span class="sensor-range">0.5 - 8.5 A</span></td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <table class="sensor-table">
                    <tr>
                        <th>Sensor Code</th>
                        <th>Description</th>
                    </tr>
                    <tr><td><strong>TP2</strong></td><td>Compressor pressure (bar)</td></tr>
                    <tr><td><strong>TP3</strong></td><td>Pneumatic panel pressure (bar)</td></tr>
                    <tr><td><strong>H1</strong></td><td>Separator pressure drop (bar)</td></tr>
                    <tr><td><strong>DV_pressure</strong></td><td>Towers discharge drop (bar)</td></tr>
                    <tr><td><strong>Reservoirs</strong></td><td>Downstream pressure (bar)</td></tr>
                    <tr><td><strong>Oil_temperature</strong></td><td>Oil temperature (°C)</td></tr>
                    <tr><td><strong>Motor_current</strong></td><td>Motor phase current (A)</td></tr>
                </table>
                """, unsafe_allow_html=True)

            st.markdown("#### Digital Sensors")
            st.markdown("""
            <table class="sensor-table">
                <tr>
                    <th>Sensor Code</th>
                    <th>Description</th>
                </tr>
                <tr><td><strong>COMP</strong></td><td>Air intake valve</td></tr>
                <tr><td><strong>DV_eletric</strong></td><td>Compressor outlet valve</td></tr>
                <tr><td><strong>Towers</strong></td><td>Tower operation selector</td></tr>
                <tr><td><strong>MPG</strong></td><td>Compressor start signal</td></tr>
                <tr><td><strong>LPS</strong></td><td>Low pressure sensor</td></tr>
                <tr><td><strong>Pressure_switch</strong></td><td>Towers discharge detector</td></tr>
                <tr><td><strong>Oil_level</strong></td><td>Oil level detector</td></tr>
                <tr><td><strong>Caudal_impulses</strong></td><td>Air flow pulse counter</td></tr>
            </table>
            """, unsafe_allow_html=True)

        # Generate data quality report first
        data_quality_report = st.session_state.sensor_validator.generate_data_quality_report(
            sensor_values, role)
        
        # Show prediction results if available
        if st.session_state.prediction_made and st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            
            # Show prediction status with enhanced visibility
            is_normal = result.get('pred_label') == 'NORMAL'
            status_color = "#10b981" if is_normal else "#ef4444"
            glow_color = "0, 255, 0" if is_normal else "255, 0, 0"
            
            animation_css = """
                <style>
                    @keyframes pulse {
                        0% { transform: scale(1); }
                        50% { transform: scale(1.02); }
                        100% { transform: scale(1); }
                    }
                    @keyframes blink {
                        0% { opacity: 1; }
                        50% { opacity: 0.6; }
                        100% { opacity: 1; }
                    }
                </style>
            """
            
            status_html = f"""
                <div style="position: relative; margin: 20px 0;">
                    <div style="
                        background: linear-gradient(145deg, {status_color}22, {status_color}44);
                        border: 2px solid {status_color};
                        border-radius: 15px;
                        padding: 20px;
                        box-shadow: 0 0 30px rgba({glow_color}, 0.3);
                        animation: pulse 2s infinite;
                    ">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="
                                width: 20px;
                                height: 20px;
                                background: {status_color};
                                border-radius: 50%;
                                animation: blink 1s infinite;
                            "></div>
                            <div>
                                <h2 style="margin: 0; color: {status_color}; font-size: 28px;">
                                    {result.get('pred_label')}
                                </h2>
                                <p style="margin: 5px 0 0 0; color: #94a3b8;">
                                    Confidence: {float(result['probability'][0]):.1%}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            """
            
            st.markdown(animation_css + status_html, unsafe_allow_html=True)

            # Display the detailed AI explanation (same for all roles)
            st.markdown("### 🧠 AI Analysis Explanation")
            
            # Create an expandable section for the full detailed analysis
            with st.expander("🤖 Detailed AI Analysis Report", expanded=True):
                st.markdown(format_explanation_text(result['explanation_text']), unsafe_allow_html=True)
            
            # Show SHAP visualization
            st.markdown("### 📈 Feature Impact Analysis (SHAP)")
            if os.path.exists(result['shap_image_path']):
                st.image(result['shap_image_path'], use_container_width=True, 
                        caption="SHAP Analysis: Most Influential Factors in Prediction")
            
            # Role-specific additional insights
            if role == "plant_manager":
                with st.expander("📊 Executive Summary - Data Quality"):
                    st.info(f"""
                    - Data Quality Score: {data_quality_report['data_quality_score']:.1f}/100
                    - Reliability Assessment: {'High confidence' if data_quality_report['data_quality_score'] > 80 else 'Requires attention'}
                    """)
                    
                    if data_quality_report['alerts']:
                        st.warning("⚠️ Additional Data Quality Alerts")
                        for alert in data_quality_report['alerts']:
                            if alert['status'] == 'critical':
                                st.markdown(f"🔴 {alert['message']}")

            elif role == "maintenance_engineer":
                with st.expander("🔧 Sensor Range Check"):
                    for param, value in result['sensor_values'].items():
                        if param in data_quality_report.get('sensor_ranges', {}):
                            ranges = data_quality_report['sensor_ranges'][param]
                            if value > ranges.get('critical_high', float('inf')):
                                st.error(f"**{param}**: {value:.2f} {ranges.get('unit', '')} - CRITICAL HIGH")
                            elif value < ranges.get('critical_low', float('-inf')):
                                st.error(f"**{param}**: {value:.2f} {ranges.get('unit', '')} - CRITICAL LOW")

            else:  # ml_engineer
                with st.expander("🤖 Model Performance Metrics"):
                    st.info(f"""
                    - Prediction Confidence: {float(result['probability'][0]):.1%}
                    - Data Quality Score: {data_quality_report['data_quality_score']:.1f}/100
                    - Feature Stability: {'Stable' if data_quality_report['data_quality_score'] > 80 else 'Requires Investigation'}
                    - Model Type: XGBoost Classifier with SHAP Explainability
                    """)

            # Show recommendations based on role
            st.markdown("### 📋 Recommendations")
            if role == "plant_manager":
                for alert in data_quality_report['alerts']:
                    if alert['status'] == 'critical':
                        st.error(f"🚨 {alert['message']}")
                    else:
                        st.warning(f"⚠️ {alert['message']}")
                    for rec in alert['recommendations']:
                        st.info(f"➡️ {rec}")
                        
            elif role == "maintenance_engineer":
                # Extract maintenance-specific recommendations
                maintenance_recs = [rec for alert in data_quality_report['alerts'] 
                                 for rec in alert['recommendations'] 
                                 if 'calibrate' in rec.lower() or 'inspect' in rec.lower()]
                for rec in maintenance_recs:
                    st.info(f"🔧 {rec}")
                    
            else:  # ml_engineer
                # Show model-specific recommendations
                if data_quality_report['data_quality_score'] < 80:
                    st.warning("⚠️ Data quality issues may affect model performance")
                    st.info("🔍 Recommend investigating feature distributions and potential sensor calibration issues")

        # Prediction results section
        if run_diagnostic:
            input_df = pd.DataFrame([sensor_values])
            
            with st.spinner("🔬 Analyzing sensor data..."):
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
                        'sensor_values': sensor_values,
                        'prediction': prediction[0],
                        'data_quality_score': data_quality_report['data_quality_score'],
                        'role': role,
                        'alerts': data_quality_report['alerts']
                    }
                    
                    # Rerun to update the display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.prediction_made = False
                    st.session_state.prediction_result = None

        # Save prediction button action
        if save_prediction and st.session_state.get('prediction_result'):
            result = st.session_state.prediction_result.copy()  # Make a copy
            
            # Convert numpy types to regular Python types
            result['prediction'] = int(result['prediction'])
            result['probability'] = [float(p) for p in result['probability']]
            result['data_quality_score'] = float(result['data_quality_score'])
            
            # Add timestamp if missing
            if 'timestamp' not in result:
                result['timestamp'] = datetime.now().isoformat()
            
            success = st.session_state.storage_manager.save_prediction(result)
            if success:
                st.success("✅ Prediction saved successfully!")
                # Clear the prediction state to prevent duplicate saves
                st.session_state.prediction_made = False
            else:
                st.error("❌ Error saving prediction")

# ========== PAGE 2: TREND ANALYSIS ==========
elif page == "Trend Analysis":
    st.markdown('<div class="section-header">📈 Historical Trend Analysis</div>', unsafe_allow_html=True)
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Time range selector for within-day analysis
    time_analysis = st.radio("Time Analysis", ["Daily Overview"])
    if time_analysis == "Within Day Analysis":
        # Use st.time_input for picking start and end time
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            start_time = st.time_input("Start Time", value=datetime.strptime("00:00", "%H:%M").time())
        with tcol2:
            end_time = st.time_input("End Time", value=datetime.strptime("23:59", "%H:%M").time())
    else:
        start_time = None
        end_time = None

    # Parameter selection for trends
    st.markdown("### 📊 Parameter Selection")
    parameters = st.multiselect(
        "Select parameters to analyze:",
        ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 
         'Oil_temperature', 'Motor_current'],
        default=['TP2', 'TP3']
    )
    
    # Get trend analysis with filters
    trend_analysis = st.session_state.storage_manager.get_trend_analysis(
        role=role,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
        time_analysis=time_analysis,
        start_time=start_time if time_analysis == "Within Day Analysis" else None,
        end_time=end_time if time_analysis == "Within Day Analysis" else None
    )

    # Display trends
    if trend_analysis['status'] == 'no_data':
        st.warning("No data available for the selected period.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Readings", trend_analysis['total_predictions'])
        with col2:
            st.metric("Failure Events", trend_analysis['failure_count'])
        with col3:
            st.metric("Failure Rate", f"{trend_analysis['failure_rate']*100:.1f}%")
        with col4:
            st.metric("Trend", trend_analysis['trend_direction'].title())
        
        # Parameter trends
        st.plotly_chart(trend_analysis['parameter_plot'], use_container_width=True)
        
        # Correlation matrix
        st.markdown("### 📊 Parameter Correlations")
        st.plotly_chart(trend_analysis['correlation_plot'], use_container_width=True)
        
        # Role-specific insights
        st.markdown(f"### 💡 Key Insights for {role.replace('_', ' ').title()}")
        for insight in trend_analysis['role_specific_insights']:
            if insight.startswith('CRITICAL'):
                st.error(insight)
            elif insight.startswith('WARNING'):
                st.warning(insight)
            else:
                st.info(insight)


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(15, 15, 35, 0.95); border-radius: 25px;">
    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Metro Train APU Health Monitor Pro v3.0 🚀
    </div>
    <div style="color: #a0a0b0; font-size: 0.9rem;">
        Powered by Explainable AI, Real-Time Validation & Predictive Analytics<br>
        <span style="color: #667eea;">Copyright © 2025 Odubiyi Ifeoluwa Antonia</span>
    </div>
</div>
""", unsafe_allow_html=True)

