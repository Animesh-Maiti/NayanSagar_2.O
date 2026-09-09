"""Standalone verification script for the AI engine dual-stage detector on demo image samples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ai_engine.infer import SonarHazardDetector


if __name__ == '__main__':
    detector = SonarHazardDetector()
    sample_dir = Path('app/static/samples')
    output_dir = Path('ai_engine/test_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(sample_dir.glob('demo_target_*.png'))
    for image_path in images:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f'Image {image_path} could not be read; skipping.')
            continue

        detections = detector.predict_dual_stage(image_bgr, conf=0.18)
        if not detections:
            stage = 'CLEAR_SEABED'
            class_name = 'CLEAR_SEABED'
            conf_score = 0.0
            shadow_detected = False
            estimated_height = 0.0
            final_status = 'CLEAR_SEABED'
        else:
            first = detections[0]
            stage = first.get('detection_type', 'SUPERVISED')
            class_name = first.get('class_name', 'unknown')
            conf_score = float(first.get('confidence', 0.0))
            shadow_detected = bool(first.get('has_shadow', False))
            estimated_height = float(first.get('estimated_height_m', 0.0))
            final_status = first.get('verification_status', 'VERIFIED_HAZARD')

        image_name = image_path.name
        print(f'{image_name} | {stage} | {class_name} | {conf_score:.2f} | {shadow_detected} | {estimated_height:.2f} | {final_status}')

        annotated = detector.draw_annotations(image_bgr, detections)
        out_path = output_dir / f"{image_path.stem}_annotated.png"
        cv2.imwrite(str(out_path), annotated)

    print(f'Wrote {len(images)} annotated verification images to {output_dir}')
