# scaler.py
import joblib

# Load the fitted scaler
scaler = joblib.load("scaler.pkl")

# Sensor features in correct order (must match training)
sensor_cols = [
    'TP2',           # Compressor pressure
    'TP3',           # Pneumatic panel pressure
    'H1',            # Pressure drop across cyclonic separator
    'DV_pressure',   # Pressure drop at towers
    'Reservoirs',    # Downstream reservoir pressure
    'Motor_current', # One phase motor current
    'Oil_temperature'# Compressor oil temperature
]

def scale_input(df):
    """Ensure consistent scaling with training pipeline"""
    return scaler.transform(df[sensor_cols])
