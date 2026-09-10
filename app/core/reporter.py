"""CSV, GeoJSON, and dashboard report generation for NayanSagar sonar anomaly records."""

from __future__ import annotations

import io
import json

import pandas as pd


def generate_anomaly_csv(detections_record):
    """Export a detections report record as a CSV string.

    Expected record fields include Target_ID, Hazard_Class, Confidence,
    Global_Pixel_Y, Global_Pixel_X, Est_Lat, Est_Lon, Est_Height_m.
    """
    if detections_record is None:
        detections_record = []
    frame = pd.DataFrame(detections_record)
    required_columns = [
        'Target_ID',
        'Hazard_Class',
        'Confidence',
        'Global_Pixel_Y',
        'Global_Pixel_X',
        'Est_Lat',
        'Est_Lon',
        'Est_Height_m',
    ]
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[required_columns]
    out = io.StringIO()
    frame.to_csv(out, index=False)
    return out.getvalue()


def generate_reports(verified_hazards_list):
    """Return CSV and GeoJSON strings for a verified hazard detection ledger.

    The returned GeoJSON payload contains features with geometry, hazard class,
    relief height, confidence, and the requested survey metadata fields.
    """
    records = verified_hazards_list or []
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame([
            {
                'Target_ID': 'NAYAN-0000',
                'Hazard_Class': 'clear_seabed',
                'Confidence': 0.0,
                'Global_Pixel_Y': 0,
                'Global_Pixel_X': 0,
                'Est_Lat': 18.92,
                'Est_Lon': 72.83,
                'Est_Height_m': 0.0,
                'Verification_Status': 'CLEAR_SEABED',
            }
        ])

    required_columns = [
        'Target_ID',
        'Hazard_Class',
        'Confidence',
        'Global_Pixel_Y',
        'Global_Pixel_X',
        'Est_Lat',
        'Est_Lon',
        'Est_Height_m',
        'Verification_Status',
    ]
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = None
    csv_frame = frame[required_columns]
    csv_io = io.StringIO()
    csv_frame.to_csv(csv_io, index=False)
    csv_text = csv_io.getvalue()

    # Build GeoJSON FeatureCollection payload.
    features = []
    for _, row in frame.iterrows():
        geometry = {
            'type': 'Point',
            'coordinates': [float(row.get('Est_Lon', 72.83)), float(row.get('Est_Lat', 18.92))],
        }
        properties = {
            'Target_ID': row.get('Target_ID', 'NAYAN-UNKNOWN'),
            'Hazard_Class': row.get('Hazard_Class', 'unknown'),
            'Confidence': float(row.get('Confidence', 0.0) or 0.0),
            'Est_Height_m': float(row.get('Est_Height_m', 0.0) or 0.0),
            'Verification_Status': row.get('Verification_Status', 'UNVERIFIED_CLUTTER'),
            'Global_Pixel_X': int(row.get('Global_Pixel_X', 0) or 0),
            'Global_Pixel_Y': int(row.get('Global_Pixel_Y', 0) or 0),
        }
        features.append({'type': 'Feature', 'geometry': geometry, 'properties': properties})

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'survey': 'NayanSagar 2.0',
            'compliance': 'MoES SIH26057 / IHO S-44',
            'physics_gate': 'Active',
        },
    }
    geojson_text = json.dumps(geojson, indent=2)

    return csv_text, geojson_text
