"""Streamlit dashboard for the NayanSagar sonar hazard detection platform."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ai_engine.infer import SonarHazardDetector
from app.core.tiler import slice_waterfall
from app.core.physics import estimate_hazard_height
from app.core.reporter import generate_anomaly_csv
from app.config import APP_CONFIG
from app.telemetry.xtf_reader import XtfReader, generate_mock_waterfall

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title='NayanSagar AI - Side-Scan Sonar Hazard Detection', layout='wide')

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #061a2d 0%, #0b3755 100%); color: #eefbff; }
    .streamlit-expanderHeader { color: #9fe2ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NayanSagar AI - Side-Scan Sonar Hazard Detection")
st.caption("Operational Status")
st.markdown("<span style='color:#00e676'>● Engine: Active</span> | <span style='color:#bbdefb'>Model: YOLO11s-seg</span>", unsafe_allow_html=True)

# Sidebar controls.
st.sidebar.header("Detection Controls")
st.sidebar.slider("Global Confidence Override", min_value=0.0, max_value=1.0, value=0.12, step=0.01, key='global_conf')
st.sidebar.checkbox("Remap Ambiguous Clutter (Crab Pots -> Debris)", value=True, key='remap_clutter')
st.sidebar.slider("Towfish Altitude Offset (m)", min_value=-5.0, max_value=50.0, value=0.0, step=0.5, key='altitude_offset')

# Example class thresholds expander.
with st.sidebar.expander("Per-Class Thresholds"):
    st.number_input("Aircraft Wreck", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key='th_aircraft')
    st.number_input("Ghost Net", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key='th_ghost')
    st.number_input("Debris Highlight", min_value=0.0, max_value=1.0, value=0.20, step=0.05, key='th_debris')
    st.number_input("Shipwreck", min_value=0.0, max_value=1.0, value=0.18, step=0.05, key='th_shipwreck')
    st.number_input("Crab Pot Trap", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key='th_crab')

# Inferencing backend.
detector = SonarHazardDetector('ai_engine/weights/best.pt')

# Tab layout.
tab1, tab2 = st.tabs(["Single Tile Inspection (.PNG / .JPG)", "Swath Telemetry Pipeline (.XTF Stream)"])

with tab1:
    col_left, col_right = st.columns(2)
    uploaded = st.file_uploader("Upload a sonar tile image (.png/.jpg)", type=['png', 'jpg', 'jpeg'])
    image = None
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)[:, :, ::-1]
    else:
        image_np = generate_mock_waterfall(rows=640, cols=640)
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    # Original vs segmented overlay.
    detections = detector.predict_tile(image_np, conf_overrides={
        'shipwreck': st.session_state.get('th_shipwreck', 0.18),
        'aircraft_wreck': st.session_state.get('th_aircraft', 0.35),
        'crab_pot_trap': st.session_state.get('th_crab', 0.15),
        'ghost_net': st.session_state.get('th_ghost', 0.30),
        'debris_highlight': st.session_state.get('th_debris', 0.20),
    }, remap_enabled=st.session_state.get('remap_clutter', True))
    overlay = detector.draw_annotations(image_np, detections)

    with col_left:
        st.subheader("Original")
        st.image(image_np[..., ::-1] if image_np.ndim == 3 else image_np, channels='BGR', use_column_width=True)
    with col_right:
        st.subheader("Segmented Mask Overlay")
        st.image(overlay[..., ::-1], channels='BGR', use_column_width=True)

    # KPI cards.
    if detections:
        confidence_values = [float(d.get('confidence', 0.0)) for d in detections]
        class_counts = pd.Series([d.get('class_name') for d in detections]).value_counts().to_dict()
        top_class = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[0][0]
        st.metric("Detections", len(detections))
        st.metric("Peak Hazard Class", top_class)
        st.metric("Mean Confidence", round(float(np.mean(confidence_values)), 4))

    # Summary table.
    df = pd.DataFrame(detections)
    if not df.empty:
        df = df[['class_name', 'confidence', 'bbox', 'contour']]
        st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Swath Telemetry Pipeline (.XTF Stream)")
    xtf_file = st.file_uploader("Load .xtf telemetry stream", type=['xtf'])
    if st.button("Load Demo Sample") or xtf_file is None:
        waterfall = generate_mock_waterfall(rows=160, cols=1000)
        st.success("Loaded demonstration synthetic waterfall sample.")
    else:
        fall_back = io.BytesIO(xtf_file.getvalue())
        # A physical file path is required for the reader; for uploaded stream store in temp.
        tmp_path = Path('data/tmp_uploaded.xtf')
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(xtf_file.getvalue())
        reader = XtfReader(tmp_path)
        waterfall, telemetry = reader.read()
    progress = st.progress(0)
    for i in range(0, 101, 20):
        progress.progress(i)

    # Render a simplified waterfall preview.
    st.image(cv2.normalize(waterfall.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), clamp=True)

    # Global location map placeholder.
    st.subheader("Global Hazard Location Plot")
    x = np.linspace(0, len(waterfall[0]) - 1, 20)
    y = np.linspace(0, len(waterfall) - 1, 20)
    global_xy = pd.DataFrame({"Global_Pixel_X": x, "Global_Pixel_Y": y})
    st.scatter_chart(global_xy)

    # CSV report download.
    sample_detections = [
        {
            'Target_ID': 'A1',
            'Hazard_Class': 'shipwreck',
            'Confidence': 0.82,
            'Global_Pixel_Y': 420,
            'Global_Pixel_X': 680,
            'Est_Lat': 12.345,
            'Est_Lon': -63.212,
            'Est_Height_m': estimate_hazard_height(12.0, 5.2, 100.0),
        }
    ]
    csv_text = generate_anomaly_csv(sample_detections)
    st.download_button('Download Anomaly CSV', csv_text, file_name='anomalies.csv', mime='text/csv')
