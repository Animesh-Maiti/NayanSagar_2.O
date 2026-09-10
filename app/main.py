"""NayanSagar Streamlit dashboard for the acoustic hazard inspection system."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import io
import logging
import tempfile
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ai_engine.infer import SonarHazardDetector
from app.core.tiler import remap_to_global_coords, slice_waterfall
from app.core.physics import estimate_hazard_height
from app.core.reporter import generate_anomaly_csv, generate_reports
from app.config import APP_CONFIG
from app.telemetry.xtf_reader import (
    XtfReader,
    generate_mock_waterfall,
    parse_xtf_or_swath,
    preprocess_acoustic_swath,
)


logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title='NayanSagar', layout='wide')
st.markdown(
    """
    <style>
    :root { --bg: #0B131E; --panel: #111B26; --line: #1E2A38; --text: #DCEAF0; --muted: #9AA8AE; --green: #98F7B6; --green-dark: #183925; --red: #C66A6A; --tile: #162233; --font: "Inter", "Segoe UI", Arial, sans-serif; }
    .stApp { background: var(--bg); color: var(--text); font-family: var(--font); }
    .main .block-container { padding-top: 1.2rem; }
    .nayan-header { display: flex; align-items: center; justify-content: space-between; padding: 0.2rem 0.6rem; border-bottom: 1px solid var(--line); background: var(--panel); color: var(--text); }
    .nayan-title { font-size: 15px; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; }
    .nayan-system { display: flex; align-items: center; gap: 8px; font-family: "Roboto Mono", "Consolas", monospace; font-size: 11px; color: var(--muted); }
    .nayan-system .dot { width: 7px; height: 7px; border-radius: 999px; background: var(--green); border: 1px solid var(--green); box-shadow: 0 0 3px var(--green); }
    .stTabs [role='tab'] { color: var(--text); background: var(--panel); border: 1px solid var(--line); border-bottom: none; font-weight: 600; }
    .stTabs [role='tab'][aria-selected='true'] { color: var(--green); border-top: 1px solid var(--green); }
    .stSidebar { background: var(--panel); border-right: 1px solid var(--line); }
    .stSidebar .block-container { padding-top: 0.6rem; }
    .stSidebar .stToggle, .stSidebar .stSlider, .stSidebar .stNumberInput, .stSidebar .stCheckbox { color: var(--text); }
    .stButton > button { border-radius: 4px; border: 1px solid var(--line); background: transparent; color: var(--text); }
    .stButton > button:hover { border-color: var(--green); color: var(--green); }
    .stDownloadButton > button { border-radius: 4px; border: 1px solid var(--green); background: var(--green-dark); color: var(--green); }
    .dataframe { border: 1px solid var(--line); }
    .report-strip { border: 1px solid var(--line); background: var(--panel); padding: 0.9rem; }
    .strip-row { display: flex; gap: 12px; align-items: center; font-family: "Roboto Mono", "Consolas", monospace; font-size: 11px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nayan-header">
      <span class="nayan-title">NayanSagar | Acoustic Anomaly & Hydrographic Clearance Console</span>
      <span class="nayan-system"><span class="dot"></span>SYSTEM: READY (OFFLINE)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('NayanSagar')
st.sidebar.markdown('### Stream Controls')

autonomous_telemetry = st.sidebar.toggle('Auto-Telemetry Link (XTF Navigation Stream)', value=True, key='autonomous_telemetry')

with st.sidebar.expander('Sensor Metrology Calibration', expanded=False):
    manual_confidence = st.slider(
        'Confidence Score',
        min_value=0.15,
        max_value=0.90,
        value=0.25,
        step=0.01,
        key='manual_confidence_input',
    )
    manual_altitude = st.number_input(
        'Towfish Altitude (m)',
        min_value=0.0,
        max_value=80.0,
        value=12.0,
        step=1.0,
        key='manual_altitude_input',
    )
    manual_slant_range = st.number_input(
        'Max Slant Range (m)',
        min_value=1.0,
        max_value=500.0,
        value=50.0,
        step=5.0,
        key='manual_slant_range_input',
    )

st.sidebar.checkbox('Noise Pre-filtering (CLAHE + Median Despeckle)', value=True, key='noise_pref_filter')
st.session_state.setdefault('active_survey_records', [])
st.session_state.setdefault('active_source_filename', 'None')
st.session_state.setdefault('last_processed_tab', None)
st.session_state.setdefault('last_tile_signature', None)

active_conf = 0.35 if autonomous_telemetry else float(manual_confidence)
active_altitude = 12.0 if autonomous_telemetry else float(manual_altitude)
active_slant_range = 50.0 if autonomous_telemetry else float(manual_slant_range)

# AI engine safe initialization.
try:
    detector = SonarHazardDetector('ai_engine/weights/best.pt')
except Exception as exc:
    detector = None
    st.warning(f'AI engine initialisation warning: {exc}')

sample_files = sorted(Path('app/static/samples').glob('demo_target_*.png'))
def load_image(file_path):
    img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((640, 640, 3), dtype=np.uint8)
    return img


def _deduplicate_global_detections(detections, center_distance_px=50.0):
    """Keep the highest-confidence detection for each nearby class cluster."""
    ranked = sorted(
        detections,
        key=lambda item: float(item.get('confidence', 0.0)),
        reverse=True,
    )
    retained = []
    for detection in ranked:
        bbox = detection.get('bbox', [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        class_name = str(detection.get('class_name', 'unknown'))
        duplicate = False
        for existing in retained:
            existing_bbox = existing['bbox']
            ex_center = (
                (float(existing_bbox[0]) + float(existing_bbox[2])) / 2.0,
                (float(existing_bbox[1]) + float(existing_bbox[3])) / 2.0,
            )
            if (
                str(existing.get('class_name', 'unknown')) == class_name
                and np.hypot(center[0] - ex_center[0], center[1] - ex_center[1]) <= center_distance_px
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append(detection)
    return retained


def _is_accepted_swath_detection(detection):
    bbox = detection.get('bbox', [0, 0, 0, 0])
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    # Broad responses are usually continuous seabed morphology rather than
    # discrete hazards, so reject detections spanning almost the full tile.
    if width > 550 or height > 550:
        return False

    # The tile center is the nadir transition; unshadowed returns there are
    # commonly boundary ringing and should not become survey candidates.
    nadir_left, nadir_right = 304, 336
    intersects_nadir = x1 <= nadir_right and x2 >= nadir_left
    if intersects_nadir and not bool(detection.get('has_shadow', False)):
        return False

    if detection.get('detection_type') != 'UNSUPERVISED_ANOMALY':
        return True
    return float(detection.get('confidence', 0.0)) >= 0.20 and width * height >= 300


def _apply_swath_physics_gate(detection, minimum_height_m=0.3):
    """Normalize final status so only shadow-backed positive relief is a hazard."""
    item = dict(detection)
    has_shadow = bool(item.get('has_shadow', False))
    height_m = float(item.get('estimated_height_m', 0.0))
    confidence = float(item.get('confidence', 0.0))
    if item.get('detection_type') == 'SUPERVISED' and confidence >= 0.50 and (
        (has_shadow and height_m >= minimum_height_m)
        or (not has_shadow and confidence >= 0.70)
    ):
        item['verification_status'] = 'VERIFIED_HAZARD'
    elif item.get('detection_type') == 'SUPERVISED' and confidence >= 0.60:
        item['verification_status'] = 'PROBABLE_HAZARD'
    else:
        item['verification_status'] = 'UNVERIFIED_CLUTTER'
    return item


def _clean_map_coordinates(frame):
    """Normalize geographic rows before passing them to Deck.gl."""
    cleaned = []
    for row in frame.to_dict('records'):
        try:
            lat = float(row.get('lat', 0.0))
            lon = float(row.get('lon', 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            lat = 18.9254 + (abs(lat) % 5000.0) / 100000.0
            lon = 72.8258 + (abs(lon) % 5000.0) / 100000.0
        lat = max(min(lat, 89.9), -89.9)
        lon = max(min(lon, 179.9), -179.9)
        if lat == 0.0 and lon == 0.0:
            continue
        cleaned.append({'lat': lat, 'lon': lon})
    return pd.DataFrame(cleaned, columns=['lat', 'lon'])


def _survey_records(detections, source_filename, metadata=None):
    records = []
    metadata_frame = metadata if isinstance(metadata, pd.DataFrame) else pd.DataFrame()
    for index, detection in enumerate(detections):
        bbox = detection.get('bbox', [0, 0, 0, 0])
        x1 = int(bbox[0]) if len(bbox) == 4 else 0
        y1 = int(bbox[1]) if len(bbox) == 4 else 0
        latitude = 18.92 + (y1 * 0.00001)
        longitude = 72.83 + (x1 * 0.00001)
        if not metadata_frame.empty:
            row_index = min(max(y1, 0), len(metadata_frame) - 1)
            row = metadata_frame.iloc[row_index]
            latitude = float(row.get('Latitude', latitude))
            longitude = float(row.get('Longitude', longitude))
        records.append({
            'Target_ID': f'NAYAN-{index + 1:04d}',
            'Source_File': source_filename,
            'Hazard_Class': detection.get('class_name', 'unknown'),
            'Confidence': round(float(detection.get('confidence', 0.0)), 4),
            'Slant_Range_m': round(float(detection.get('slant_range_m', 0.0)), 2),
            'Estimated_Height_m': round(float(detection.get('estimated_height_m', 0.0)), 2),
            'Verification_Status': (
                'VERIFIED_HAZARD'
                if bool(detection.get('has_shadow', False))
                and float(detection.get('estimated_height_m', 0.0)) >= 0.25
                else (
                    'PROBABLE_HAZARD'
                    if detection.get('detection_type') == 'SUPERVISED'
                    and float(detection.get('confidence', 0.0)) >= 0.60
                    else 'UNVERIFIED_CLUTTER'
                )
            ),
            'BBox_Coordinates': str(detection.get('bbox', [])),
            'Latitude': round(latitude, 6),
            'Longitude': round(longitude, 6),
            'Timestamp_UTC': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ'),
        })
    if not records:
        records.append({
            'Target_ID': 'NAYAN-CLEAR-0001',
            'Source_File': source_filename,
            'Hazard_Class': 'clear_seabed',
            'Confidence': 1.0,
            'Estimated_Height_m': 0.0,
            'Slant_Range_m': 0.0,
            'Verification_Status': 'SURVEY_CLEARED_NO_OBSTRUCTION',
            'Latitude': 18.92,
            'Longitude': 72.83,
            'Timestamp_UTC': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ'),
        })
    return records


def _active_reports(records):
    frame = pd.DataFrame(records)
    csv_text = frame.to_csv(index=False)
    features = []
    for record in records:
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [float(record.get('Longitude', 72.83)), float(record.get('Latitude', 18.92))],
            },
            'properties': {
                key: value for key, value in record.items()
                if key not in {'Latitude', 'Longitude'}
            },
        })
    return csv_text, json.dumps({
        'type': 'FeatureCollection',
        'features': features,
    }, indent=2)


# Duplicate the requested three-tab UI.
tab_quick, tab_swath, tab_export = st.tabs([
    'Tile Analysis',
    'Swath Telemetry',
    'Survey Archive',
])

with tab_quick:
    st.markdown('### Tile Analysis')
    sample_name = st.selectbox('Choose a preloaded side-scan sample', [p.name for p in sample_files], index=0)
    uploaded = st.file_uploader('Upload a sonar tile image (.png/.jpg)', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

    if uploaded is not None:
        raw_img = Image.open(uploaded).convert('RGB')
        image_bgr = np.array(raw_img)[:, :, ::-1]
    else:
        image_bgr = load_image(Path('app/static/samples') / sample_name)

    # Optional preprocessing pipeline for quick scan
    if st.session_state.get('noise_pref_filter', True):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        filt = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(filt)
        image_bgr_pre = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr_pre = image_bgr.copy()

    conf = active_conf
    towfish_altitude = active_altitude
    slant_range = active_slant_range
    if detector is not None:
        detections = detector.predict_dual_stage(
            image_bgr_pre,
            conf=conf,
            conf_thresh=active_conf,
            towfish_altitude_m=towfish_altitude,
            slant_range_m=slant_range,
            meters_per_pixel=(2.0 * slant_range / max(1, image_bgr_pre.shape[1])),
        )
    else:
        detections = []

    if detector is not None:
        try:
            annotation = detector.draw_annotations(image_bgr_pre, detections)
        except Exception:
            annotation = image_bgr_pre
    else:
        annotation = image_bgr_pre
    current_filename = uploaded.name if uploaded is not None else sample_name
    tile_signature = (
        current_filename,
        int(getattr(uploaded, 'size', 0)) if uploaded is not None else None,
    )
    if tile_signature != st.session_state.get('last_tile_signature'):
        st.session_state['active_source_filename'] = current_filename
        st.session_state['active_survey_records'] = _survey_records(
            detections,
            current_filename,
        )
        st.session_state['last_processed_tab'] = 'tab1'
        st.session_state['last_tile_signature'] = tile_signature

    left, right = st.columns(2)
    with left:
        st.caption('Acoustic Return (Raw)')
        st.image(image_bgr_pre[:, :, ::-1], channels='BGR', width='stretch')
    with right:
        st.caption('Target Clearance Analysis')
        st.image(annotation[:, :, ::-1], channels='BGR', width='stretch')

    hazards = detections if detections else []
    table = []
    for d in hazards:
        table.append({
            'Target Classification': d.get('class_name', 'uncataloged_hazard'),
            'Slant-Range Position': f'{float(d.get("slant_range_m", slant_range)):.2f}m',
            'Shadow Relief Height': f'{float(d.get("estimated_height_m", 0.0)):.2f}m',
            'Confidence Score': float(d.get('confidence', 0.0)),
        })
    if table:
        telemetry_df = pd.DataFrame(table)
    else:
        telemetry_df = pd.DataFrame([
            {
                'Target Classification': 'CLEAR_SEABED',
                'Slant-Range Position': f'{slant_range:.1f}m',
                'Shadow Relief Height': '0.00m',
                'Confidence Score': 0.0,
            }
        ])
    st.markdown('<div class="report-strip"><div class="strip-row">', unsafe_allow_html=True)
    st.dataframe(telemetry_df, hide_index=True, width='stretch')
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('### Survey Telemetry Log')
    st.dataframe(pd.DataFrame({
        'Target Classification': [d.get('class_name', 'uncataloged_hazard') for d in hazards],
        'Slant-Range Position': [f'{float(d.get("slant_range_m", slant_range)):.2f}m' for d in hazards],
        'Shadow Relief Height': [f'{float(d.get("estimated_height_m", 0.0)):.2f}m' for d in hazards],
        'Confidence Score': [float(d.get('confidence', 0.0)) for d in hazards],
    }) if hazards else pd.DataFrame([
        {'Target Classification': 'CLEAR_SEABED', 'Slant-Range Position': f'{slant_range:.1f}m', 'Shadow Relief Height': '0.00m', 'Confidence Score': 0.0}
    ]), hide_index=True, width='stretch')

with tab_swath:
    st.markdown('### Swath Telemetry')
    upload = st.file_uploader('Upload XTF or acoustic swath image', type=['xtf', 'png', 'jpg', 'jpeg'])
    if upload is not None:
        try:
            raw_waterfall, metadata = parse_xtf_or_swath(upload)
        except Exception as exc:
            raw_waterfall = generate_mock_waterfall(rows=240, cols=1000)
            metadata = pd.DataFrame([])
            st.warning(f'Unable to parse uploaded data directly; falling back to synthetic swath: {exc}')
    else:
        raw_waterfall = generate_mock_waterfall(rows=240, cols=1000)
        metadata = pd.DataFrame([
            {
                'PingID': 1,
                'PortChannel': 0,
                'StarboardChannel': 1,
                'Latitude': 18.92,
                'Longitude': 72.83,
                'Altitude_m': 12.0,
                'SlantRange_m': 50.0,
                'Pitch_deg': 0.0,
                'Roll_deg': 0.0,
            }
        ])

    if st.session_state.get('autonomous_telemetry', True):
        active_altitude = (
            float(metadata['Altitude_m'].dropna().iloc[0])
            if 'Altitude_m' in metadata.columns and metadata['Altitude_m'].notna().any()
            else 12.0
        )
        active_slant_range = (
            float(metadata['SlantRange_m'].dropna().iloc[0])
            if 'SlantRange_m' in metadata.columns and metadata['SlantRange_m'].notna().any()
            else 50.0
        )

    st.subheader('Acoustic Waterfall')
    raw_col, proc_col = st.columns(2)
    display_width = min(1600, max(640, raw_waterfall.shape[1]))
    display_height = max(1, int(round(raw_waterfall.shape[0] * display_width / max(1, raw_waterfall.shape[1]))))
    raw_display = cv2.resize(
        raw_waterfall.astype(np.uint8),
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )
    raw_col.image(raw_display, clamp=True, caption='Acoustic Return (Raw)')
    preproc = preprocess_acoustic_swath(raw_waterfall, altitude_px=40)
    proc_display = cv2.resize(
        preproc.astype(np.uint8),
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )
    proc_col.image(proc_display, clamp=True, caption='Processed Waterfall')

    st.subheader('Swath Processing Queue')
    preview_left, preview_right = st.columns(2)
    raw_preview = preview_left.empty()
    annotated_preview = preview_right.empty()
    progress = st.progress(0)
    progress_label = st.empty()
    slices = slice_waterfall(preproc, tile_size=640, overlap=128)
    total_slices = len(slices)
    tiles_analyzed = []
    st.session_state['swath_slice_cache'] = {}
    for i, (tile, x_offset, y_offset, slice_id) in enumerate(slices, start=1):
        progress_label.caption(f'Processing slice {i}/{total_slices}...')
        if detector is not None:
            try:
                swath_conf_thresh = 0.55
                tile_conf = swath_conf_thresh
                tile_altitude = active_altitude
                tile_range = active_slant_range
                tile_dets = detector.predict_dual_stage(
                    tile,
                    conf=tile_conf,
                    conf_thresh=swath_conf_thresh,
                    towfish_altitude_m=tile_altitude,
                    slant_range_m=tile_range,
                    meters_per_pixel=(2.0 * tile_range / max(1, tile.shape[1])),
                )
                accepted_tile_dets = [
                    detection for detection in tile_dets
                    if float(detection.get('confidence', 0.0)) >= 0.45
                    and _is_accepted_swath_detection(detection)
                    and (
                        detection.get('detection_type') != 'SUPERVISED'
                        or float(detection.get('confidence', 0.0)) >= swath_conf_thresh
                    )
                ]
                accepted_tile_dets = [
                    _apply_swath_physics_gate(detection)
                    for detection in accepted_tile_dets
                ]
                global_dets = remap_to_global_coords(accepted_tile_dets, (y_offset, x_offset))
                if global_dets:
                    raw_preview.image(
                        tile,
                        channels='GRAY',
                        clamp=True,
                        caption=f'Raw Swath Slice [ID: {slice_id}]',
                    )
                    if detector is not None:
                        annotated_tile = detector.draw_annotations(tile.copy(), accepted_tile_dets)
                    else:
                        annotated_tile = tile
                    annotated_preview.image(
                        annotated_tile,
                        channels='BGR' if annotated_tile.ndim == 3 else 'GRAY',
                        clamp=True,
                        caption=f'Active Target Clearance Analysis [Detections: {len(global_dets)}]',
                    )
                    st.session_state['swath_slice_cache'][slice_id] = {
                        'raw_tile': tile.copy(),
                        'annotated_tile': annotated_tile.copy(),
                        'detections': [dict(detection) for detection in accepted_tile_dets],
                    }
                for detection in global_dets:
                    detection['slice_id'] = slice_id
                tiles_analyzed.extend(global_dets)
            except Exception:
                tile_dets = []
        progress.progress(int(min(100, 100 * i / max(1, total_slices))))
    progress_label.caption(f'Processed {total_slices} slices.')
    tiles_analyzed = _deduplicate_global_detections(tiles_analyzed)
    verified_detections = [
        detection for detection in tiles_analyzed
        if detection.get('verification_status') == 'VERIFIED_HAZARD'
    ]
    clutter_detections = [
        detection for detection in tiles_analyzed
        if detection.get('verification_status') != 'VERIFIED_HAZARD'
        and float(detection.get('confidence', 0.0)) >= 0.45
    ]
    if upload is not None:
        source_filename = upload.name
        st.session_state['active_source_filename'] = source_filename
        st.session_state['active_survey_records'] = _survey_records(
            tiles_analyzed,
            source_filename,
            metadata,
        )
        st.session_state['last_processed_tab'] = 'tab2'

    st.success(f'Processed {len(slices)} swath slices with {len(verified_detections)} verified hazards.')
    swath_table = pd.DataFrame([
        {
            'Slice': detection.get('slice_id', ''),
            'Class': detection.get('class_name', 'unknown'),
            'Stage': detection.get('detection_type', ''),
            'Confidence': float(detection.get('confidence', 0.0)),
            'Shadow': bool(detection.get('has_shadow', False)),
            'Height_m': float(detection.get('estimated_height_m', 0.0)),
            'SlantRange_m': float(detection.get('slant_range_m', active_slant_range)),
            'Status': detection.get('verification_status', ''),
        }
        for detection in verified_detections
    ])
    if swath_table.empty:
        swath_table = pd.DataFrame([{'Status': 'CLEAR_SEABED'}])
    st.dataframe(swath_table, hide_index=True, width='stretch')
    with st.expander('Show Raw Acoustic Clutter Candidates (Telemetry Noise Filtered)'):
        clutter_table = pd.DataFrame([
            {
                'Slice': detection.get('slice_id', ''),
                'Class': detection.get('class_name', 'unknown'),
                'Stage': detection.get('detection_type', ''),
                'Confidence': float(detection.get('confidence', 0.0)),
                'Shadow': bool(detection.get('has_shadow', False)),
                'Height_m': float(detection.get('estimated_height_m', 0.0)),
                'Status': 'UNVERIFIED_CLUTTER',
            }
            for detection in clutter_detections
        ])
        if clutter_table.empty:
            clutter_table = pd.DataFrame([{'Status': 'NO_FILTERED_CLUTTER'}])
        st.dataframe(clutter_table, hide_index=True, width='stretch')

    st.markdown('### Swath Slice Detail Inspector')
    slice_cache = st.session_state.get('swath_slice_cache', {})
    if slice_cache:
        selected_slice = st.selectbox(
            'Select Slice ID to Inspect:',
            options=list(slice_cache.keys()),
        )
        selected = slice_cache[selected_slice]
        inspect_left, inspect_right = st.columns(2)
        inspect_left.image(
            selected['raw_tile'],
            clamp=True,
            caption=f'Raw Slice Return: {selected_slice}',
        )
        inspect_right.image(
            selected['annotated_tile'],
            channels='BGR' if selected['annotated_tile'].ndim == 3 else 'GRAY',
            clamp=True,
            caption=f'Physics Verification: {selected_slice}',
        )
        st.dataframe(pd.DataFrame([
            {
                'Class': detection.get('class_name', 'unknown'),
                'Confidence': float(detection.get('confidence', 0.0)),
                'Shadow Length (px)': float(detection.get('shadow_length_px', 0.0)),
                'Estimated Height (m)': float(detection.get('estimated_height_m', 0.0)),
                'Slant Range (m)': float(detection.get('slant_range_m', active_slant_range)),
                'Status': detection.get('verification_status', ''),
            }
            for detection in selected['detections']
        ]), hide_index=True, width='stretch')
    else:
        st.caption('No accepted detections are available for slice inspection.')

    st.subheader('Georeferenced Survey Track')
    base_lat = float(metadata['Latitude'].dropna().iloc[0]) if 'Latitude' in metadata.columns and metadata['Latitude'].notna().any() else 18.92
    base_lon = float(metadata['Longitude'].dropna().iloc[0]) if 'Longitude' in metadata.columns and metadata['Longitude'].notna().any() else 72.83
    raw_map_points = pd.DataFrame({
        'lat': [base_lat + float(d.get('bbox', [0, 0, 0, 0])[1]) * 1e-6 for d in verified_detections] or [base_lat],
        'lon': [base_lon + float(d.get('bbox', [0, 0, 0, 0])[0]) * 1e-6 for d in verified_detections] or [base_lon],
    })
    map_points = _clean_map_coordinates(raw_map_points)
    if map_points.empty:
        st.info('No georeferenced coordinates available to plot on map.')
    else:
        st.map(map_points)


with tab_export:
    st.markdown('### Survey Archive')
    session_ledger = st.session_state.get('active_survey_records', [])
    st.subheader('Survey Inspection Log')
    if not session_ledger:
        st.info('No active survey telemetry available. Run Tile Analysis or upload an XTF swath to compile hydrographic records.')
    else:
        source_filename = st.session_state.get('active_source_filename', 'None')
        st.caption(f'Active Survey Dataset: **{source_filename}** | Total Hazards Logged: {len(session_ledger)}')
        ledger_frame = pd.DataFrame(session_ledger)
        st.dataframe(ledger_frame, width='stretch')
        csv_report, geojson_report = _active_reports(session_ledger)
        filename_stem = Path(source_filename).stem or 'survey'
        st.subheader('Reports')
        st.download_button(
            'Export IHO S-44 Survey Log (CSV)',
            csv_report,
            file_name=f'NayanSagar_IHO_S44_Log_{filename_stem}.csv',
            mime='text/csv',
        )
        st.download_button(
            'Export Spatial Vector (GeoJSON)',
            geojson_report,
            file_name=f'NayanSagar_Hazards_{filename_stem}.geojson',
            mime='application/geo+json',
        )
