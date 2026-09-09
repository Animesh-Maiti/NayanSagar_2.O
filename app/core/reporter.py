"""CSV report generation for exported sonar anomaly records."""

from __future__ import annotations

import io

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
