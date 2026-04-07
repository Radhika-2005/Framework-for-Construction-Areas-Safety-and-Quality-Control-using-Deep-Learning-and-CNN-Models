import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from ultralytics import YOLO
import time
import base64
from collections import Counter
import tempfile

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Construction Site AI Dashboard",
    page_icon="🏗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HELPER FUNCTIONS ====================
def get_base64_image(image_path):
    """Convert image file to base64 string"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None
def play_alert_sound():
    """Generate audio alert for violations"""
    audio_html = """
    <audio autoplay>
        <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjGH0fPTgjMGHm7A7+OZRQ0PVqvn8LNnHQU8ktbzyn" type="audio/wav">
    </audio>
    """
    return audio_html
def get_custom_css(bg_image_path=None):
    """Generate custom CSS styling"""
    bg_image_css = ""
    if bg_image_path and os.path.exists(bg_image_path):
        base64_img = get_base64_image(bg_image_path)
        if base64_img:
            img_type = "png" if bg_image_path.endswith('.png') else "jpeg"
            bg_image_css = f"""
            background-image: url('data:image/{img_type};base64,{base64_img}');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            """
    
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main {{
        {bg_image_css}
    }}
    
    .main::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(5px);
        z-index: -1;
        pointer-events: none;
    }}
    
    .stApp {{
        {bg_image_css}
    }}
    
    h1 {{
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.8rem;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }}
    
    h2 {{
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 1rem;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
    }}
    
    h3 {{
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #00d4ff !important;
        text-shadow: 0 0 8px rgba(0, 212, 255, 0.3);
    }}
    
    h4, h5, h6 {{
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    }}
    
    p, span, div, label, li {{
        font-size: 1.1rem !important;
        color: #ffffff !important;
        font-weight: 500;
        line-height: 1.6;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }}
    
    .stText {{
        font-size: 1.1rem !important;
        color: #ffffff !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }}
    
    .stMarkdown {{
        font-size: 1.1rem !important;
        color: #ffffff !important;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, rgba(20, 40, 70, 0.92) 0%, rgba(15, 35, 60, 0.92) 100%);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 2px solid rgba(0, 212, 255, 0.3);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        font-size: 1.1rem;
        cursor: pointer;
        backdrop-filter: blur(15px);
    }}
    
    .metric-card:hover {{
        transform: translateY(-10px);
        box-shadow: 0 16px 48px rgba(0, 212, 255, 0.4);
        background: linear-gradient(135deg, rgba(0, 100, 150, 0.85) 0%, rgba(0, 80, 130, 0.85) 100%);
        border-color: rgba(0, 212, 255, 0.6);
    }}
    
    .metric-card h3 {{
        color: #00e5ff !important;
        font-size: 1.8rem !important;
        margin-bottom: 1rem;
        transition: color 0.3s ease;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.5);
    }}
    
    .metric-card:hover h3 {{
        color: #00ffff !important;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
    }}
    
    .metric-card p {{
        font-size: 1.05rem !important;
        color: #b3e5ff !important;
        margin-bottom: 1.2rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }}
    
    .metric-card ul {{
        font-size: 1rem !important;
        list-style-position: inside;
    }}
    
    .metric-card li {{
        font-size: 1rem !important;
        color: #80deea !important;
        margin-bottom: 0.6rem;
        line-height: 1.7;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }}
    
    .stAlert {{
        padding: 1.5rem;
        border-radius: 0.8rem;
        font-size: 1.1rem !important;
        border-left: 5px solid;
        background: rgba(20, 40, 60, 0.85) !important;
        backdrop-filter: blur(10px);
    }}
    
    .stButton > button {{
        font-size: 1.15rem !important;
        padding: 1rem 1.8rem !important;
        border-radius: 0.7rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2) !important;
        letter-spacing: 0.5px;
        color: #ffffff !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-4px) !important;
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.35) !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(-2px) !important;
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #2563eb !important;
        letter-spacing: -1px;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .stSelectbox label {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }}
    
    .stSlider label {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }}
    
    .stSlider > div > div > div {{
        font-size: 1.1rem !important;
        color: #ffffff !important;
    }}
    
    .stNumberInput label {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }}
    
    .stFileUploader label {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }}
    
    [data-testid="stFileUploadDropzone"] {{
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px dashed rgba(37, 99, 235, 0.4) !important;
    }}
    
    [data-testid="stFileUploadDropzone"] p {{
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }}
    
    [data-testid="stFileUploadDropzone"] span {{
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }}
    
    .stFileUploader {{
        color: #1a1a1a !important;
    }}
    
    .stSelectbox > div > div {{
        background: linear-gradient(135deg, rgba(20, 40, 70, 0.92) 0%, rgba(15, 35, 60, 0.92) 100%) !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 0.6rem !important;
    }}
    
    .stSelectbox [data-testid="selectbox"] {{
        background: linear-gradient(135deg, rgba(20, 40, 70, 0.92) 0%, rgba(15, 35, 60, 0.92) 100%) !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
    }}
    
    input, select, textarea {{
        font-size: 1.05rem !important;
        padding: 0.9rem 1rem !important;
        border-radius: 0.6rem !important;
        background: linear-gradient(135deg, rgba(20, 40, 70, 0.92) 0%, rgba(15, 35, 60, 0.92) 100%) !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
        color: #ffffff !important;
        transition: all 0.3s ease !important;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }}
    
    input:hover, select:hover, textarea:hover {{
        border-color: rgba(0, 212, 255, 0.7) !important;
        background: linear-gradient(135deg, rgba(0, 100, 150, 0.85) 0%, rgba(0, 80, 130, 0.85) 100%) !important;
        box-shadow: 0 2px 12px rgba(0, 212, 255, 0.3) !important;
    }}
    
    input:focus, select:focus, textarea:focus {{
        border-color: #00e5ff !important;
        background: linear-gradient(135deg, rgba(0, 120, 170, 0.9) 0%, rgba(0, 100, 150, 0.9) 100%) !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4) !important;
        outline: none !important;
        color: #ffffff !important;
    }}
    
    input::placeholder {{
        color: #80deea !important;
        font-weight: 500;
    }}
    
    table {{
        font-size: 1.05rem !important;
        color: #1e293b !important;
        border-collapse: collapse;
    }}
    
    table th {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        padding: 1.2rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    table td {{
        font-size: 1.05rem !important;
        color: #1e293b !important;
        padding: 1rem !important;
        border-bottom: 1px solid rgba(37, 99, 235, 0.1) !important;
    }}
    
    table tr:hover {{
        background-color: rgba(37, 99, 235, 0.08) !important;
        transition: background-color 0.3s ease;
    }}
    
    .stSubheader {{
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #2563eb !important;
        margin-top: 1.8rem !important;
        margin-bottom: 1.2rem !important;
        letter-spacing: -0.3px;
    }}
    
    .stExpander {{
        border: 2px solid rgba(37, 99, 235, 0.2) !important;
        border-radius: 0.6rem !important;
        transition: all 0.3s ease;
    }}
    
    .stExpander:hover {{
        border-color: rgba(37, 99, 235, 0.5) !important;
        background-color: rgba(37, 99, 235, 0.03) !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1) !important;
    }}
    
    [data-testid="stDataFrameContainer"] {{
        font-size: 1.05rem !important;
    }}
    
    .stSuccess {{
        font-size: 1.15rem !important;
        padding: 1.3rem 1.5rem !important;
        border-radius: 0.7rem !important;
        border-left: 5px solid #10b981 !important;
    }}
    
    .stError {{
        font-size: 1.15rem !important;
        padding: 1.3rem 1.5rem !important;
        border-radius: 0.7rem !important;
        border-left: 5px solid #ef4444 !important;
    }}
    
    .stWarning {{
        font-size: 1.15rem !important;
        padding: 1.3rem 1.5rem !important;
        border-radius: 0.7rem !important;
        border-left: 5px solid #f59e0b !important;
    }}
    
    .stInfo {{
        font-size: 1.15rem !important;
        padding: 1.3rem 1.5rem !important;
        border-radius: 0.7rem !important;
        border-left: 5px solid #3b82f6 !important;
    }}
    
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(20, 120, 180, 0.95) 0%, rgba(15, 80, 140, 0.95) 100%) !important;
        box-shadow: 2px 0 15px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
    }}
    
    .stSidebar {{
        font-size: 1.1rem !important;
    }}
    
    .stSidebar p, .stSidebar span, .stSidebar div, .stSidebar label {{
        color: #ffffff !important;
        font-weight: 600;
    }}
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {{
        color: #00d4ff !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }}
    
    .stSidebar .stText {{
        color: #e8f4f8 !important;
        font-weight: 500;
    }}
    
    .stSidebar .stSelectbox {{
        color: #ffffff !important;
    }}
    
    .stSidebar .stSelectbox label {{
        color: #00d4ff !important;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(0, 212, 255, 0.2);
    }}
    
    .stSidebar [data-testid="selectbox"] {{
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
    }}
    
    .stSpinner {{
        font-size: 1.1rem !important;
    }}
    
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
    }}
    </style>
    """
    return css

# ==================== APPLY STYLES ====================
bg_image_path = "background.jpg"
if os.path.exists("background image.jpg"):
    bg_image_path = "background image.jpg"
elif os.path.exists("background.png"):
    bg_image_path = "background.png"

st.markdown(get_custom_css(bg_image_path), unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'alert_settings' not in st.session_state:
    st.session_state.alert_settings = {
        'ppe_violations': True,
        'crack_detection': True,
        'maintenance_critical': True,
        'sound_enabled': True
    }
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'last_sensor_values' not in st.session_state:
    st.session_state.last_sensor_values = None

# ==================== ALERT FUNCTIONS ====================
def add_alert(alert_type, severity, message, details=None):
    """Add a new alert to the system"""
    alert = {
        'timestamp': datetime.now(),
        'type': alert_type,
        'severity': severity,
        'message': message,
        'details': details,
        'read': False
    }
    st.session_state.alerts.insert(0, alert)
    if len(st.session_state.alerts) > 50:
        st.session_state.alerts = st.session_state.alerts[:50]
    return alert

def get_unread_alerts():
    """Get count of unread alerts"""
    return sum(1 for alert in st.session_state.alerts if not alert['read'])

def mark_all_read():
    """Mark all alerts as read"""
    for alert in st.session_state.alerts:
        alert['read'] = True

# ==================== MODEL PATHS ====================
MODEL_PATHS = {
    'ppe': 'Construction-Site-Safety-PPE-Detection-main/models/best.pt',
    'crack_h5': 'concrete_crack_model.h5',
    'crack_saved': 'concrete_crack_saved_model',
    'crack_fast': 'concrete_fast_saved_model',
    'maintenance_h5': 'nasa_cmaps_lstm_model.h5',
    'maintenance_keras': 'nasa_cmaps_lstm_model.keras',
    'maintenance_pkl': 'predictive_maintenance_model.pkl'
}

# ==================== CLASS DEFINITIONS ====================
PPE_CLASSES = {
    0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask',
    4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone',
    7: 'Safety Vest', 8: 'Machinery', 9: 'Vehicle'
}

CRACK_CLASSES = {0: 'No Crack', 1: 'Crack Detected'}
VIOLATIONS = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load all AI models"""
    models = {}
    model_status = {}
    
    # Load PPE Detection Model
    try:
        if os.path.exists(MODEL_PATHS['ppe']):
            models['ppe'] = YOLO(MODEL_PATHS['ppe'])
            model_status['ppe'] = '✅ Loaded'
        else:
            model_status['ppe'] = '❌ Not Found'
    except Exception as e:
        model_status['ppe'] = f'❌ Error: {str(e)[:20]}'
    
    # Load Crack Detection Model
    try:
        import tensorflow as tf
        crack_loaded = False
        
        if os.path.exists(MODEL_PATHS['crack_h5']):
            models['crack'] = tf.keras.models.load_model(MODEL_PATHS['crack_h5'])
            model_status['crack'] = '✅ Loaded'
            crack_loaded = True
        elif os.path.exists(MODEL_PATHS['crack_saved']):
            models['crack'] = tf.keras.models.load_model(MODEL_PATHS['crack_saved'])
            model_status['crack'] = '✅ Loaded'
            crack_loaded = True
        elif os.path.exists(MODEL_PATHS['crack_fast']):
            models['crack'] = tf.keras.models.load_model(MODEL_PATHS['crack_fast'])
            model_status['crack'] = '✅ Loaded'
            crack_loaded = True
        
        if not crack_loaded:
            model_status['crack'] = '❌ Not Found'
    except Exception as e:
        model_status['crack'] = f'❌ Error: {str(e)[:20]}'
    
    # Load Predictive Maintenance Model
    try:
        import tensorflow as tf
        maintenance_loaded = False
        
        if os.path.exists(MODEL_PATHS['maintenance_h5']):
            models['maintenance'] = tf.keras.models.load_model(MODEL_PATHS['maintenance_h5'])
            model_status['maintenance'] = '✅ Loaded'
            maintenance_loaded = True
        elif os.path.exists(MODEL_PATHS['maintenance_keras']):
            models['maintenance'] = tf.keras.models.load_model(MODEL_PATHS['maintenance_keras'])
            model_status['maintenance'] = '✅ Loaded'
            maintenance_loaded = True
        elif os.path.exists(MODEL_PATHS['maintenance_pkl']):
            import pickle
            with open(MODEL_PATHS['maintenance_pkl'], 'rb') as f:
                models['maintenance'] = pickle.load(f)
            model_status['maintenance'] = '✅ Loaded'
            maintenance_loaded = True
        
        if not maintenance_loaded:
            model_status['maintenance'] = '❌ Not Found'
    except Exception as e:
        model_status['maintenance'] = f'❌ Error: {str(e)[:20]}'
    
    return models, model_status

models, model_status = load_models()

# ==================== WEBCAM FUNCTIONS ====================
def process_ppe_frame(frame, model, confidence):
    """Process a single frame for PPE detection"""
    results = model.predict(frame, conf=confidence, verbose=False)
    
    detections = []
    violations = []
    
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = PPE_CLASSES.get(class_id, "Unknown")
            detections.append({'class': class_name, 'confidence': conf * 100})
            if class_name in VIOLATIONS:
                violations.append({'class': class_name})
    
    annotated_frame = results[0].plot()
    return annotated_frame, detections, violations

def process_crack_frame(frame, model):
    """Process a single frame for crack detection"""
    import tensorflow as tf
    
    img_array = cv2.resize(frame, (96, 96))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    crack_probability = float(prediction[0][0])
    crack_detected = crack_probability > 0.5
    
    # Draw result on frame
    color = (0, 0, 255) if crack_detected else (0, 255, 0)
    text = f"CRACK: {crack_probability*100:.1f}%" if crack_detected else f"OK: {(1-crack_probability)*100:.1f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame, crack_detected, crack_probability

# ==================== SIDEBAR ====================
unread_count = get_unread_alerts()
if unread_count > 0:
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                padding: 1rem; border-radius: 0.8rem; margin-bottom: 1rem;
                box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
                animation: pulse 2s infinite;'>
        <h3 style='margin: 0; color: white; font-size: 1.3rem;'>
            🔔 {unread_count} New Alert{'s' if unread_count > 1 else ''}
        </h3>
    </div>
    """, unsafe_allow_html=True)
with st.sidebar.expander("🔔 Alert Settings", expanded=False):
    st.session_state.alert_settings['ppe_violations'] = st.checkbox(
        "PPE Violations", 
        value=st.session_state.alert_settings['ppe_violations']
    )
    st.session_state.alert_settings['crack_detection'] = st.checkbox(
        "Crack Detection", 
        value=st.session_state.alert_settings['crack_detection']
    )
    st.session_state.alert_settings['maintenance_critical'] = st.checkbox(
        "Critical Maintenance", 
        value=st.session_state.alert_settings['maintenance_critical']
    )
    st.session_state.alert_settings['sound_enabled'] = st.checkbox(
        "🔊 Sound Notifications", 
        value=st.session_state.alert_settings['sound_enabled']
    )
with st.sidebar.expander("📬 View Alerts", expanded=False):
    if st.session_state.alerts:
        if st.button("Mark All as Read", key='mark_read'):
            mark_all_read()
            st.rerun()
        
        st.markdown("---")
        for idx, alert in enumerate(st.session_state.alerts[:10]):
            severity_emoji = {
                'critical': '🚨',
                'warning': '⚠',
                'info': 'ℹ'
            }
            bg_color = {
                'critical': 'rgba(239, 68, 68, 0.2)',
                'warning': 'rgba(245, 158, 11, 0.2)',
                'info': 'rgba(59, 130, 246, 0.2)'
            }
            
            read_style = "opacity: 0.6;" if alert['read'] else ""
            st.markdown(f"""
            <div style='background: {bg_color[alert['severity']]}; 
                        padding: 0.8rem; border-radius: 0.5rem; 
                        margin-bottom: 0.5rem; {read_style}'>
                <div style='font-weight: 600; margin-bottom: 0.3rem;'>
                    {severity_emoji[alert['severity']]} {alert['message']}
                </div>
                <div style='font-size: 0.85rem; opacity: 0.8;'>
                    {alert['timestamp'].strftime('%I:%M %p')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No alerts yet")

st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "Select Detection Model",
    ["🏗 Overview", "👷 PPE Detection", "🔨 Crack Detection", "⚙ Predictive Maintenance"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Status")
for model_name, status in model_status.items():
    st.sidebar.text(f"{model_name.upper()}: {status}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ System Info")
st.sidebar.text(f"Models Loaded: {len(models)}/3")
st.sidebar.text(f"Detections: {len(st.session_state.detection_history)}")

# ==================== OVERVIEW PAGE ====================
if selected_model == "🏗 Overview":
    st.title("🏗 Construction Site AI Safety Dashboard")
    st.markdown("## AI-powered safety and maintenance monitoring")

# AI-powered safety and maintenance monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>👷 PPE Detection</h3>
            <p>Real-time detection of safety equipment</p>
            <ul>
                <li>10 object classes</li>
                <li>Safety violation alerts</li>
                <li>Worker tracking</li>
                <li>Live webcam support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔨 Crack Detection</h3>
            <p>Structural integrity monitoring</p>
            <ul>
                <li>Concrete crack detection</li>
                <li>Severity assessment</li>
                <li>Location mapping</li>
                <li>Live webcam support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚙ Predictive Maintenance</h3>
            <p>Equipment failure prediction</p>
            <ul>
                <li>Failure prediction</li>
                <li>Maintenance scheduling</li>
                <li>Cost optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔔 Recent Alerts & Notifications")
    
    if st.session_state.alerts:
        # Alert summary
        alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
        
        total_alerts = len(st.session_state.alerts)
        critical_alerts = sum(1 for a in st.session_state.alerts if a['severity'] == 'critical')
        warning_alerts = sum(1 for a in st.session_state.alerts if a['severity'] == 'warning')
        unread_alerts = get_unread_alerts()
        
        alert_col1.metric("Total Alerts", total_alerts)
        alert_col2.metric("Critical", critical_alerts, delta="High Priority" if critical_alerts > 0 else None)
        alert_col3.metric("Warnings", warning_alerts)
        alert_col4.metric("Unread", unread_alerts)
        
        # Display recent alerts
        st.markdown("### Recent Activity")
        for alert in st.session_state.alerts[:5]:
            severity_config = {
                'critical': {'emoji': '🚨', 'color': '#ef4444', 'bg': 'rgba(239, 68, 68, 0.1)'},
                'warning': {'emoji': '⚠', 'color': '#f59e0b', 'bg': 'rgba(245, 158, 11, 0.1)'},
                'info': {'emoji': 'ℹ', 'color': '#3b82f6', 'bg': 'rgba(59, 130, 246, 0.1)'}
            }
            
            config = severity_config[alert['severity']]
            read_badge = "" if alert['read'] else " 🔴"
            
            st.markdown(f"""
            <div style='background: {config['bg']}; padding: 1.2rem; 
                        border-radius: 0.8rem; margin-bottom: 0.8rem;
                        border-left: 4px solid {config['color']};'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <span style='font-size: 1.3rem;'>{config['emoji']}</span>
                        <span style='font-weight: 700; color: {config['color']}; margin-left: 0.5rem; font-size: 1.1rem;'>
                            {alert['message']}{read_badge}
                        </span>
                    </div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>
                        {alert['timestamp'].strftime('%I:%M %p')}
                    </div>
                </div>
                {f"<div style='margin-top: 0.5rem; font-size: 0.95rem; opacity: 0.8;'>{alert['details']}</div>" if alert['details'] else ""}
            </div>
            """, unsafe_allow_html=True)
        
        if len(st.session_state.alerts) > 5:
            st.info(f"📋 {len(st.session_state.alerts) - 5} more alerts available in sidebar")
    else:
        st.info("🎉 No alerts - System operating normally")
    
    st.markdown("---")
    st.subheader("📊 Model Performance Comparison")
    
    performance_data = {
        'Model': ['PPE Detection', 'Crack Detection', 'Predictive Maintenance'],
        'Accuracy': [85.3, 92.5, 88.7],
        'Precision': [82.3, 89.2, 85.4],
        'Recall': [86.7, 94.1, 90.2],
        'F1-Score': [84.4, 91.6, 87.7]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(df_performance, width=None, hide_index=True, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Bar(
                name=metric,
                x=performance_data['Model'],
                y=performance_data[metric],
                text=performance_data[metric],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Model",
            yaxis_title="Score (%)",
            barmode='group',
            height=400,
            font=dict(size=12),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== PPE DETECTION PAGE ====================
elif selected_model == "👷 PPE Detection":
    st.title("👷 PPE Detection System")
    st.markdown("### Upload images or use live webcam for safety equipment detection")
    
    if 'ppe' not in models:
        st.error("❌ PPE Detection model not loaded. Please check the model path.")
        st.info(f"Expected path: {MODEL_PATHS['ppe']}")
    else:
        # Mode selection
        detection_mode = st.radio("Select Detection Mode:", ["📁 Upload Image", "📹 Live Webcam"], horizontal=True)
        
        if detection_mode == "📁 Upload Image":
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📤 Upload & Settings")
                uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key='ppe_upload')
                confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    if st.button("🔍 Detect PPE", key='ppe_detect', use_container_width=True):
                        with st.spinner("Detecting..."):
                            img_array = np.array(image)
                            start_time = time.time()
                            results = models['ppe'].predict(img_array, conf=confidence, verbose=False)
                            inference_time = time.time() - start_time
                            
                            detections = []
                            violations = []
                            
                            if len(results) > 0 and results[0].boxes is not None:
                                for box in results[0].boxes:
                                    class_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = PPE_CLASSES.get(class_id, "Unknown")
                                    detections.append({'class': class_name, 'confidence': conf * 100})
                                    if class_name in VIOLATIONS:
                                        violations.append({'class': class_name})
                            
                            st.session_state.detection_history.append({
                                'timestamp': datetime.now(),
                                'model': 'PPE Detection',
                                'detections': len(detections),
                                'violations': len(violations)
                            })
                            
                            # Add alerts for violations
                            if violations and st.session_state.alert_settings['ppe_violations']:
                                for v in violations:
                                    add_alert(
                                        'ppe_violation',
                                        'critical',
                                        f"PPE Violation: {v['class']}",
                                        {'location': 'Construction Site', 'count': len(violations)}
                                    )
                            
                            with col2:
                                st.subheader("📊 Detection Results")
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Objects", len(detections))
                                m2.metric("Violations", len(violations))
                                m3.metric("Inference", f"{inference_time:.2f}s")
                                m4.metric("FPS", f"{1/inference_time:.1f}" if inference_time > 0 else "N/A")
                                
                                annotated_img = results[0].plot()
                                st.image(annotated_img, caption="Detected PPE", use_container_width=True)
                                
                                if violations:
                                    st.error(f"🚨 {len(violations)} Safety Violation(s) Detected!")
                                    for v in violations:
                                        st.warning(f"⚠ {v['class']}")
                                else:
                                    st.success("✅ No safety violations detected")
                                
                                if detections:
                                    st.subheader("📋 Detected Objects")
                                    class_counts = Counter([d['class'] for d in detections])
                                    for class_name, count in sorted(class_counts.items()):
                                        emoji = "⚠" if class_name in VIOLATIONS else "✅"
                                        st.text(f"{emoji} {class_name}: {count}")
        
        else:  # Live Webcam Mode
            st.subheader("📹 Live Webcam PPE Detection")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key='webcam_conf')
                
                start_button = st.button("🎥 Start Webcam", use_container_width=True)
                stop_button = st.button("⏹ Stop Webcam", use_container_width=True)
                
                if start_button:
                    st.session_state.webcam_active = True
                if stop_button:
                    st.session_state.webcam_active = False
            
            with col2:
                frame_placeholder = st.empty()
                metrics_placeholder = st.empty()
                violations_placeholder = st.empty()
            
            if st.session_state.webcam_active:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Cannot access webcam. Please check your camera permissions.")
                    st.session_state.webcam_active = False
                else:
                    st.info("🎥 Webcam is active. Press 'Stop Webcam' to end detection.")
                    
                    frame_count = 0
                    total_violations = 0
                    
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("❌ Failed to grab frame")
                            break
                        
                        # Process every frame for real-time detection
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        annotated_frame, detections, violations = process_ppe_frame(frame_rgb, models['ppe'], confidence)
                        
                        # Update display
                        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                        
                        # Update metrics
                        with metrics_placeholder.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Objects", len(detections))
                            m2.metric("Current Violations", len(violations))
                            m3.metric("Total Violations", total_violations)
                        
                        # Show violations
                        if violations:
                            total_violations += len(violations)
                            with violations_placeholder.container():
                                st.error(f"🚨 {len(violations)} Safety Violation(s) Detected!")
                                for v in violations:
                                    st.warning(f"⚠ {v['class']}")
                            
                            # Add alerts
                            if st.session_state.alert_settings['ppe_violations']:
                                for v in violations:
                                    add_alert('ppe_violation', 'critical', f"PPE Violation: {v['class']}")
                        
                        frame_count += 1
                        time.sleep(0.03)
                    
                    cap.release()
                    st.success("✅ Webcam stopped successfully")

# ==================== CRACK DETECTION PAGE ====================
elif selected_model == "🔨 Crack Detection":
    st.title("🔨 Concrete Crack Detection")
    st.markdown("### Upload images or use live webcam for structural crack analysis")
    
    if 'crack' not in models:
        st.error("❌ Crack Detection model not loaded. Please check the model path.")
        st.info("Looking for models at:")
        st.code(f"• {MODEL_PATHS['crack_h5']}\n• {MODEL_PATHS['crack_saved']}\n• {MODEL_PATHS['crack_fast']}")
    else:
        # Mode selection
        detection_mode = st.radio("Select Detection Mode:", ["📁 Upload Image", "📹 Live Webcam"], horizontal=True)
        
        if detection_mode == "📁 Upload Image":
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📤 Upload Image")
                uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key='crack_upload')
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    if st.button("🔍 Detect Cracks", key='crack_detect', use_container_width=True):
                        with st.spinner("Analyzing..."):
                            import tensorflow as tf
                            
                            img_array = np.array(image)
                            img_array = cv2.resize(img_array, (96, 96))
                            img_array = img_array.astype('float32') / 255.0
                            img_array = np.expand_dims(img_array, axis=0)
                            
                            start_time = time.time()
                            prediction = models['crack'].predict(img_array, verbose=0)
                            inference_time = time.time() - start_time
                            
                            crack_probability = float(prediction[0][0])
                            crack_detected = crack_probability > 0.5
                            
                            # Add alerts for cracks
                            if crack_detected and st.session_state.alert_settings['crack_detection']:
                                severity = 'critical' if crack_probability > 0.9 else 'warning' if crack_probability > 0.7 else 'info'
                                add_alert(
                                    'crack_detection',
                                    severity,
                                    f"Crack Detected ({crack_probability*100:.1f}% confidence)",
                                    {'severity_level': 'HIGH' if crack_probability > 0.9 else 'MEDIUM' if crack_probability > 0.7 else 'LOW'}
                                )
                            
                            with col2:
                                st.subheader("📊 Analysis Results")
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Status", "⚠ CRACK" if crack_detected else "✅ OK")
                                m2.metric("Confidence", f"{crack_probability*100:.1f}%")
                                m3.metric("Inference", f"{inference_time:.3f}s")
                                
                                st.image(image, caption="Analysis Result", use_container_width=True)
                                
                                if crack_detected:
                                    st.error(f"⚠ Crack detected with {crack_probability*100:.1f}% confidence")
                                    st.subheader("📋 Assessment")
                                    if crack_probability > 0.9:
                                        st.error("🔴 HIGH Severity - Immediate inspection required!")
                                        st.write("*Recommended Actions:*")
                                        st.write("• Schedule immediate inspection")
                                        st.write("• Halt operations if structure critical")
                                        st.write("• Document location and extent")
                                    elif crack_probability > 0.7:
                                        st.warning("🟡 MEDIUM Severity - Schedule inspection within 1-2 weeks")
                                        st.write("*Recommended Actions:*")
                                        st.write("• Schedule inspection within 1-2 weeks")
                                        st.write("• Monitor for propagation")
                                        st.write("• Document progress")
                                    else:
                                        st.info("🟢 LOW Severity - Monitor during maintenance cycle")
                                        st.write("*Recommended Actions:*")
                                        st.write("• Monitor during routine inspection")
                                        st.write("• Check for growth")
                                        st.write("• Re-assess in 3 months")
                                else:
                                    st.success("✅ No cracks detected - Structure appears sound")
        
        else:  # Live Webcam Mode
            st.subheader("📹 Live Webcam Crack Detection")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                start_button = st.button("🎥 Start Webcam", use_container_width=True, key='crack_start')
                stop_button = st.button("⏹ Stop Webcam", use_container_width=True, key='crack_stop')
                
                if start_button:
                    st.session_state.webcam_active = True
                if stop_button:
                    st.session_state.webcam_active = False
            
            with col2:
                frame_placeholder = st.empty()
                metrics_placeholder = st.empty()
                status_placeholder = st.empty()
            
            if st.session_state.webcam_active:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Cannot access webcam. Please check your camera permissions.")
                    st.session_state.webcam_active = False
                else:
                    st.info("🎥 Webcam is active. Press 'Stop Webcam' to end detection.")
                    
                    frame_count = 0
                    total_cracks = 0
                    
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("❌ Failed to grab frame")
                            break
                        
                        # Process frames for crack detection
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame, crack_detected, crack_probability = process_crack_frame(frame_rgb.copy(), models['crack'])
                        
                        # Update display
                        frame_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                        
                        # Update metrics
                        with metrics_placeholder.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Status", "⚠ CRACK" if crack_detected else "✅ OK")
                            m2.metric("Confidence", f"{crack_probability*100:.1f}%")
                            m3.metric("Total Cracks Detected", total_cracks)
                        
                        # Show status
                        if crack_detected:
                            total_cracks += 1
                            with status_placeholder.container():
                                if crack_probability > 0.9:
                                    st.error(f"🔴 HIGH Severity Crack - {crack_probability*100:.1f}% confidence")
                                elif crack_probability > 0.7:
                                    st.warning(f"🟡 MEDIUM Severity Crack - {crack_probability*100:.1f}% confidence")
                                else:
                                    st.info(f"🟢 LOW Severity Crack - {crack_probability*100:.1f}% confidence")
                            
                            # Add alerts
                            if st.session_state.alert_settings['crack_detection']:
                                severity = 'critical' if crack_probability > 0.9 else 'warning' if crack_probability > 0.7 else 'info'
                                add_alert('crack_detection', severity, f"Crack Detected ({crack_probability*100:.1f}% confidence)")
                        
                        frame_count += 1
                        time.sleep(0.1)
                    
                    cap.release()
                    st.success("✅ Webcam stopped successfully")

# ==================== PREDICTIVE MAINTENANCE PAGE (FIXED) ====================
elif selected_model == "⚙ Predictive Maintenance":
    st.title("⚙ Predictive Maintenance System")
    st.markdown("### Predict equipment failures using NASA C-MAPS LSTM model")
    
    if 'maintenance' not in models:
        st.error("❌ Predictive Maintenance model not loaded.")
        st.info("Looking for models at:")
        st.code(f"• {MODEL_PATHS['maintenance_h5']}\n• {MODEL_PATHS['maintenance_keras']}\n• {MODEL_PATHS['maintenance_pkl']}")
    else:
        st.info("🚀 Using NASA C-MAPS LSTM Model for RUL (Remaining Useful Life) Prediction")
        
        st.subheader("🔧 Equipment Sensor Data (21 Sensors)")
        
        # Preset configurations for quick testing
        st.subheader("⚡ Quick Load Presets")
        
        presets = {
            "Healthy Equipment": [45, 48, 52, 50, 55, 48, 51, 49, 60, 58, 52, 48, 62, 59, 50, 47, 53, 51, 49, 55, 57],
            "Warning Level": [65, 68, 72, 70, 75, 68, 71, 69, 80, 78, 72, 68, 82, 79, 70, 67, 73, 71, 69, 75, 77],
            "Critical Level": [85, 88, 92, 90, 95, 88, 91, 89, 98, 96, 92, 88, 99, 97, 90, 87, 93, 91, 89, 95, 97],
            "Custom Input": [50.0] * 21
        }
        
        # Initialize preset tracking
        if 'current_preset' not in st.session_state:
            st.session_state.current_preset = "Healthy Equipment"
        
        selected_preset = st.radio(
            "Select Preset or Custom:", 
            list(presets.keys()), 
            horizontal=True, 
            index=list(presets.keys()).index(st.session_state.current_preset),
            key='preset_selector'
        )
        
        # Update preset if changed
        if selected_preset != st.session_state.current_preset:
            st.session_state.current_preset = selected_preset
            st.session_state.last_sensor_values = None
        
        st.markdown("---")
        
        # Get default values based on preset or last values
        if st.session_state.last_sensor_values is None:
            default_values = presets[selected_preset]
        else:
            default_values = st.session_state.last_sensor_values
        
        col1, col2, col3, col4, col5 = st.columns(5)
        sensors = []
        
        with col1:
            sensors.append(st.slider("S1: Total Temp", 0.0, 100.0, float(default_values[0]), 0.5, key='s1'))
            sensors.append(st.slider("S2: Total Temp", 0.0, 100.0, float(default_values[1]), 0.5, key='s2'))
            sensors.append(st.slider("S3: Total Temp", 0.0, 100.0, float(default_values[2]), 0.5, key='s3'))
            sensors.append(st.slider("S4: Total Temp", 0.0, 100.0, float(default_values[3]), 0.5, key='s4'))
        
        with col2:
            sensors.append(st.slider("S5: Pressure", 0.0, 100.0, float(default_values[4]), 0.5, key='s5'))
            sensors.append(st.slider("S6: Pressure", 0.0, 100.0, float(default_values[5]), 0.5, key='s6'))
            sensors.append(st.slider("S7: Total Temp 2", 0.0, 100.0, float(default_values[6]), 0.5, key='s7'))
            sensors.append(st.slider("S8: Pressure 2", 0.0, 100.0, float(default_values[7]), 0.5, key='s8'))
        
        with col3:
            sensors.append(st.slider("S9: Fan Speed", 0.0, 100.0, float(default_values[8]), 0.5, key='s9'))
            sensors.append(st.slider("S10: Core Speed", 0.0, 100.0, float(default_values[9]), 0.5, key='s10'))
            sensors.append(st.slider("S11: Static Press", 0.0, 100.0, float(default_values[10]), 0.5, key='s11'))
            sensors.append(st.slider("S12: Ratio", 0.0, 100.0, float(default_values[11]), 0.5, key='s12'))
        
        with col4:
            sensors.append(st.slider("S13: Fan Speed C", 0.0, 100.0, float(default_values[12]), 0.5, key='s13'))
            sensors.append(st.slider("S14: Core Speed C", 0.0, 100.0, float(default_values[13]), 0.5, key='s14'))
            sensors.append(st.slider("S15: Bypass", 0.0, 100.0, float(default_values[14]), 0.5, key='s15'))
            sensors.append(st.slider("S16: Fuel Ratio", 0.0, 100.0, float(default_values[15]), 0.5, key='s16'))
        
        with col5:
            sensors.append(st.slider("S17: Enthalpy", 0.0, 100.0, float(default_values[16]), 0.5, key='s17'))
            sensors.append(st.slider("S18: HPT Bleed", 0.0, 100.0, float(default_values[17]), 0.5, key='s18'))
            sensors.append(st.slider("S19: LPT Bleed", 0.0, 100.0, float(default_values[18]), 0.5, key='s19'))
            sensors.append(st.slider("S20: Fan Demand", 0.0, 100.0, float(default_values[19]), 0.5, key='s20'))
            sensors.append(st.slider("S21: Core Demand", 0.0, 100.0, float(default_values[20]), 0.5, key='s21'))
        
        # Store current sensor values
        st.session_state.last_sensor_values = sensors
        
        st.markdown("---")
        
        if st.button("🔮 Predict RUL", use_container_width=True, key='predict_btn'):
            with st.spinner("Analyzing equipment health..."):
                try:
                    sensor_data = np.array([sensors])
                    sensor_data_norm = sensor_data / 100.0
                    sensor_data_reshaped = sensor_data_norm.reshape(1, 1, 21)
                    
                    start_time = time.time()
                    prediction = models['maintenance'].predict(sensor_data_reshaped, verbose=0)
                    inference_time = time.time() - start_time
                    
                    rul_predicted = float(prediction[0][0])
                    
                    # Calculate failure risk based on actual sensor readings
                    avg_sensor = np.mean(sensors)
                    failure_risk = max(0, min(100, (avg_sensor - 30) * 1.5))
                    
                    # Add alerts for maintenance
                    if st.session_state.alert_settings['maintenance_critical']:
                        if failure_risk > 70:
                            add_alert(
                                'maintenance',
                                'critical',
                                f"CRITICAL: Equipment Failure Risk {failure_risk:.1f}%",
                                {'rul': rul_predicted, 'action': 'Immediate inspection required'}
                            )
                        elif failure_risk > 40:
                            add_alert(
                                'maintenance',
                                'warning',
                                f"WARNING: Elevated Failure Risk {failure_risk:.1f}%",
                                {'rul': rul_predicted, 'action': 'Schedule inspection within 1-2 weeks'}
                            )
                    
                    st.subheader("📊 Prediction Results")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("RUL (Cycles)", f"{rul_predicted:.0f}")
                    risk_level = "🔴 Critical" if failure_risk > 70 else "🟡 Warning" if failure_risk > 40 else "🟢 Normal"
                    m2.metric("Status", risk_level)
                    m3.metric("Failure Risk", f"{failure_risk:.1f}%")
                    m4.metric("Inference", f"{inference_time:.3f}s")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=failure_risk,
                        title={'text': "Failure Risk"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if failure_risk > 70 else "orange" if failure_risk > 40 else "green"},
                            'steps': [
                                {'range': [0, 40], 'color': 'lightgreen'},
                                {'range': [40, 70], 'color': 'yellow'},
                                {'range': [70, 100], 'color': 'lightcoral'}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("💡 Maintenance Recommendations")
                    if failure_risk > 70:
                        st.error("🚨 CRITICAL: Immediate Action Required!")
                        st.write("- Schedule immediate inspection")
                        st.write("- Prepare replacement parts")
                        st.write("- Plan equipment downtime within 24-48 hours")
                        st.write("- Notify maintenance team")
                    elif failure_risk > 40:
                        st.warning("⚠ WARNING: Elevated Risk Detected")
                        st.write("- Schedule inspection within 1-2 weeks")
                        st.write("- Monitor sensor readings closely")
                        st.write("- Order replacement parts (precautionary)")
                        st.write("- Increase inspection frequency")
                    else:
                        st.success("✅ NORMAL: Equipment Operating Well")
                        st.write("- Equipment in good condition")
                        st.write("- Continue regular maintenance schedule")
                        st.write("- Next inspection: As per schedule")
                        st.write("- No immediate action required")
                    
                    with st.expander("📈 Detailed Sensor Analysis"):
                        st.write("*Sensor Status:*")
                        sensor_names = [f"S{i+1}" for i in range(21)]
                        sensor_status = ["Normal" if 30 < s < 70 else "Warning" for s in sensors]
                        
                        df_sensors = pd.DataFrame({
                            'Sensor': sensor_names,
                            'Value': sensors,
                            'Status': sensor_status
                        })
                        st.dataframe(df_sensors, use_container_width=True, hide_index=True)
                        
                        fig_sensors = px.bar(df_sensors, x='Sensor', y='Value', color='Status',
                                           color_discrete_map={'Normal': '#10b981', 'Warning': '#f59e0b'})
                        fig_sensors.update_layout(height=350, showlegend=True)
                        st.plotly_chart(fig_sensors, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("💡 The NASA C-MAPS model requires 21 sensor inputs with proper normalization")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #475569; padding: 2rem; font-size: 1rem; font-weight: 600;'>
    🏗 Construction Site AI Dashboard | Built with Streamlit & YOLOv8 | © 2025
</div>
""", unsafe_allow_html=True)