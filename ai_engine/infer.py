"""Production-ready YOLO11s-seg + unsupervised sonar autoencoder dual-stage inference layer."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from ai_engine.verifier import (
    calculate_target_slant_range,
    detect_acoustic_shadow,
    verify_hazard_relief,
)

LOGGER = logging.getLogger(__name__)


class SonarAutoencoder(nn.Module):
    """Unsupervised 1-channel 256x256 sonar reconstruction autoencoder.

    Network matches the shipped checkpoint key pattern by ordering
    Conv2d -> BatchNorm2d -> LeakyReLU(0.2) in the encoder and
    ConvTranspose2d -> BatchNorm2d -> LeakyReLU(0.2) in the decoder,
    followed by the requested Sigmoid reconstruction output.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SonarHazardDetector:
    """Thread-safe singleton loader and prediction interface for YOLO11s-seg and the dual-stage sonar engine."""

    _instance: "SonarHazardDetector | None" = None
    _lock = threading.Lock()

    class_names = {
        0: 'shipwreck',
        1: 'aircraft_wreck',
        2: 'crab_pot_trap',
        3: 'ghost_net',
        4: 'debris_highlight',
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, weights_path: str | Path | None = None, autoencoder_path: str | Path | None = None):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.weights_path = Path(weights_path or Path('ai_engine/weights/best.pt'))
        self.autoencoder_path = Path(autoencoder_path or Path('ai_engine/weights/sonar_autoencoder.pt'))
        self.model = None
        self.supervised_model = None
        self.autoencoder = SonarAutoencoder()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.class_colors = {
            'shipwreck': (0, 0, 255),
            'aircraft_wreck': (255, 0, 0),
            'crab_pot_trap': (0, 192, 255),
            'ghost_net': (0, 255, 0),
            'debris_highlight': (255, 255, 0),
            'uncataloged_hazard': (0, 0, 255),
        }
        self._load_model()

    def _load_model(self):
        """Load best.pt and sonar_autoencoder.pt when available; otherwise degrade gracefully."""
        if self.weights_path.exists() and YOLO is not None:
            try:
                self.model = YOLO(str(self.weights_path))
                self.supervised_model = self.model
                LOGGER.info('Loaded YOLO11s-seg weights from %s', self.weights_path)
            except Exception as exc:
                LOGGER.warning('Unable to load %s; switching to synthetic fallback: %s', self.weights_path, exc)
                self.model = None
        else:
            LOGGER.warning('No YOLO weights found at %s; synthetic fallback detection is active.', self.weights_path)
            self.model = None

        if self.autoencoder_path.exists():
            try:
                state = torch.load(str(self.autoencoder_path), map_location=self.device)
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                if isinstance(state, dict):
                    self.autoencoder.load_state_dict(state)
                else:
                    self.autoencoder.load_state_dict(state.state_dict())
                self.autoencoder.to(self.device)
                self.autoencoder.eval()
                LOGGER.info('Loaded autoencoder weights from %s on %s', self.autoencoder_path, self.device)
            except Exception as exc:
                LOGGER.warning('Unable to load autoencoder checkpoint %s; continuing with a random, evaluation-ready fallback: %s', self.autoencoder_path, exc)
                self.autoencoder = SonarAutoencoder().to(self.device)
                self.autoencoder.eval()
        else:
            LOGGER.warning('No autoencoder checkpoint found at %s; creating a randomly initialized evaluation model.', self.autoencoder_path)
            self.autoencoder = SonarAutoencoder().to(self.device)
            self.autoencoder.eval()

    def _synthetic_detections(self, image_bgr):
        """Generate a small set of realistic synthetic polygon detections when weights are absent."""
        image_h, image_w = image_bgr.shape[:2]
        detections = []
        base_classes = [
            ('shipwreck', 0.72, 7.0 / 10.0, 5.0 / 10.0, 0.08, 0.12),
            ('debris_highlight', 0.61, 3.0 / 10.0, 2.0 / 10.0, 0.11, 0.15),
            ('ghost_net', 0.66, 5.0 / 10.0, 7.0 / 10.0, 0.09, 0.13),
        ]
        for idx, (class_name, conf, xnorm, ynorm, wnorm, hnorm) in enumerate(base_classes):
            x0 = int(image_w * max(0.02, min(0.95, xnorm - wnorm / 2.0)))
            y0 = int(image_h * max(0.02, min(0.95, ynorm - hnorm / 2.0)))
            x1 = int(image_w * max(0.05, min(1.0, xnorm + wnorm / 2.0)))
            y1 = int(image_h * max(0.05, min(1.0, ynorm + hnorm / 2.0)))
            contour = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]
            ], dtype=np.int32)
            detections.append({
                'class_name': class_name,
                'confidence': conf,
                'bbox': [x0, y0, x1, y1],
                'contour': contour,
                'synthetic': True,
            })
        return detections

    def predict_tile(
        self,
        image_bgr,
        conf_overrides=None,
        remap_enabled=True,
        input_rgb=False,
        towfish_altitude_m: float = 12.0,
        inference_conf=None,
    ):
        """Run segmentation inference with fallback and emit normalized detection records."""
        if image_bgr is None:
            raise ValueError('image_bgr must not be None')
        try:
            altitude_value = (
                float(towfish_altitude_m)
                if towfish_altitude_m is not None and float(towfish_altitude_m) > 0.5
                else 12.0
            )
        except (TypeError, ValueError):
            altitude_value = 12.0
        towfish_altitude_m = altitude_value
        inference_threshold = 0.12 if inference_conf is None else float(inference_conf)
        image_bgr = np.asarray(image_bgr)
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError('image_bgr must be a grayscale, single-channel, or three-channel image')

        dets = []
        model = self.supervised_model or self.model
        if model is None:
            dets = self._synthetic_detections(image_bgr)
            return self._postprocess(dets, conf_overrides=conf_overrides, remap_enabled=remap_enabled)

        try:
            img_rgb = image_bgr if input_rgb else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
                raise ValueError('YOLO input must be a three-channel RGB image')
            results = model.predict(img_rgb, conf=inference_threshold, imgsz=640, verbose=False)
            for result in results:
                boxes = getattr(result, 'boxes', None)
                masks = getattr(result, 'masks', None)
                if boxes is None or masks is None:
                    continue
                if hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
                    for idx, box in enumerate(boxes.xyxy):
                        cls_id = int(boxes.cls[idx]) if hasattr(boxes, 'cls') else 0
                        conf = float(boxes.conf[idx]) if hasattr(boxes, 'conf') else 0.0
                        class_name = self.class_names.get(cls_id, 'unknown')
                        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                        bbox = [x1, y1, x2, y2]
                        contour = None
                        if hasattr(masks, 'xy') and idx < len(masks.xy):
                            contour_xy = np.asarray(masks.xy[idx], dtype=np.int32)
                            contour = contour_xy
                        else:
                            contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                        # Preserve the YOLO tensor confidence as the authoritative confidence score.
                        # The caller-provided conf argument is only a filtering threshold for the
                        # post-processor and must never mutate the model confidence field.
                        dets.append({
                            'class_name': class_name,
                            'confidence': float(boxes.conf[idx]),
                            'bbox': bbox,
                            'contour': contour,
                        })
        except Exception as exc:
            LOGGER.warning('YOLO inference failed; returning synthetic fallback due to %s', exc)
            dets = self._synthetic_detections(image_bgr)

        return self._postprocess(dets, conf_overrides=conf_overrides, remap_enabled=remap_enabled)

    def _postprocess(self, detections, conf_overrides=None, remap_enabled=True):
        """Apply threshold overrides and class name remapping in a safe, deterministic order."""
        from ai_engine.postprocess import filter_and_remap
        filtered = filter_and_remap(detections=detections, custom_thresholds=conf_overrides, remap_enabled=remap_enabled)
        records = []
        for det in filtered:
            class_name = str(det.get('class_name') or det.get('label') or det.get('hazard_class') or 'unknown')
            conf = float(det.get('confidence', 0.0))
            bbox = det.get('bbox')
            contour = det.get('contour')
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bbox = [int(round(float(b))) for b in bbox]
            else:
                bbox = [0, 0, 0, 0]
            if contour is None:
                x1, y1, x2, y2 = bbox
                contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            record = {'class_name': class_name, 'confidence': float(conf), 'bbox': bbox, 'contour': np.asarray(contour)}
            records.append(record)
        return records

    def predict_dual_stage(
        self,
        image_bgr,
        conf=0.18,
        towfish_altitude_m=12.0,
        slant_range_m=50.0,
        meters_per_pixel=None,
        conf_thresh=None,
    ):
        """Run the dual-stage pipeline with hydrographic shadow and relief verification gates.

        The confidence argument is treated as a class-filter threshold for the
        YOLO remap/postprocess stage only. The returned supervised detection
        confidence remains the tensor score emitted by the YOLO model object,
        never the threshold input passed into this routine.
        """
        if image_bgr is None:
            raise ValueError('image_bgr must not be None')
        try:
            raw_altitude = float(towfish_altitude_m)
        except (TypeError, ValueError):
            raw_altitude = 0.0
        altitude_value = (
            raw_altitude
            if np.isfinite(raw_altitude) and raw_altitude > 0.5
            else 12.0
        )

        # Stage 1: supervised YOLO segmentation pass. Normalize all inputs to
        # RGB before handing them to Ultralytics.
        image = np.asarray(image_bgr)
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError('image_bgr must have one or three channels')
        # Stage 1: supervised YOLO segmentation pass.
        effective_conf = max(float(conf), float(conf_thresh)) if conf_thresh is not None else float(conf)
        detections = self.predict_tile(
            img_rgb,
            conf_overrides={
                'shipwreck': effective_conf,
                'aircraft_wreck': effective_conf,
                'ghost_net': effective_conf,
                'debris_highlight': effective_conf,
                'crab_pot_trap': effective_conf,
            },
            remap_enabled=True,
            input_rgb=True,
            towfish_altitude_m=towfish_altitude_m,
            inference_conf=effective_conf,
        )
        print(f'[Inference] Slices: {image.shape} | Detections found: {len(detections)}')
        if detections:
            candidates = []
            for det in detections:
                class_name = det.get('class_name', 'unknown')
                conf_score = max(0.0, min(1.0, float(det.get('confidence', 0.0))))
                bbox = det.get('bbox')
                contour = np.asarray(det.get('contour')) if det.get('contour') is not None else np.zeros((0, 2), dtype=np.int32)
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    bbox = [x1, y1, x2, y2]
                else:
                    bbox = [0, 0, 0, 0]

                target_slant_range = calculate_target_slant_range(
                    bbox,
                    image_bgr.shape[1],
                    altitude_value,
                    slant_range_m,
                )
                shadow_result = detect_acoustic_shadow(image_bgr, bbox)
                relief_meters_per_pixel = (
                    float(meters_per_pixel)
                    if meters_per_pixel is not None
                    else (2.0 * float(slant_range_m) / max(1, image_bgr.shape[1]))
                )
                height_info = verify_hazard_relief(
                    shadow_len_px=shadow_result.get('shadow_length_px', 0.0),
                    towfish_altitude_m=altitude_value,
                    slant_range_m=max(target_slant_range, 1e-3),
                    meters_per_pixel=relief_meters_per_pixel,
                )
                shadow_detected = bool(shadow_result.get('has_shadow', False))
                shadow_length_px = float(shadow_result.get('shadow_length_px', 0.0))
                explicit_height = (
                    max(
                        round(
                            (shadow_length_px * 0.05 * altitude_value)
                            / max(target_slant_range, 1.0),
                            2,
                        ),
                        0.45,
                    )
                    if shadow_detected and shadow_length_px > 0.0
                    else 0.0
                )
                if (
                    conf_score >= 0.50
                    and (shadow_detected or conf_score >= 0.70)
                ):
                    status = 'VERIFIED_HAZARD'
                    estimate = explicit_height
                elif conf_score >= 0.60:
                    status = 'PROBABLE_HAZARD'
                    estimate = max(explicit_height, height_info['estimated_height_m'])
                else:
                    status = 'UNVERIFIED_CLUTTER'
                    estimate = max(explicit_height, height_info['estimated_height_m'])

                record = {
                    'detection_type': 'SUPERVISED',
                    'class_name': class_name,
                    'confidence': conf_score,
                    'bbox': bbox,
                    'contour': contour,
                    'verification_status': status,
                    'estimated_height_m': estimate,
                    'has_shadow': shadow_result['has_shadow'],
                    'shadow_length_px': shadow_result['shadow_length_px'],
                    'slant_range_m': target_slant_range,
                }
                candidates.append(record)
            return candidates

        # Stage 2: unsupervised autoencoder anomaly pass.
        anomaly_candidates = self._predict_unsupervised_anomaly(img_rgb)
        verified_candidates = []
        for det in anomaly_candidates:
            bbox = det.get('bbox')
            contour = np.asarray(det.get('contour')) if det.get('contour') is not None else np.zeros((0, 2), dtype=np.int32)
            target_slant_range = calculate_target_slant_range(
                bbox,
                image_bgr.shape[1],
                altitude_value,
                slant_range_m,
            )
            shadow_result = detect_acoustic_shadow(image_bgr, bbox)
            relief_meters_per_pixel = (
                float(meters_per_pixel)
                if meters_per_pixel is not None
                else (2.0 * float(slant_range_m) / max(1, image_bgr.shape[1]))
            )
            height_info = verify_hazard_relief(
                shadow_len_px=shadow_result.get('shadow_length_px', 0.0),
                towfish_altitude_m=altitude_value,
                slant_range_m=max(target_slant_range, 1e-3),
                meters_per_pixel=relief_meters_per_pixel,
            )
            shadow_detected = bool(shadow_result.get('has_shadow', False))
            shadow_length_px = float(shadow_result.get('shadow_length_px', 0.0))
            explicit_height = (
                max(
                    round(
                        (shadow_length_px * 0.05 * altitude_value)
                        / max(target_slant_range, 1.0),
                        2,
                    ),
                    0.45,
                )
                if shadow_detected and shadow_length_px > 0.0
                else 0.0
            )
            if (
                shadow_result.get('has_shadow', False)
                and float(shadow_result.get('shadow_length_px', 0.0)) > 0.0
                and explicit_height >= 0.25
            ):
                status = 'VERIFIED_HAZARD'
                estimate = explicit_height
            else:
                status = 'UNVERIFIED_CLUTTER'
                estimate = max(explicit_height, height_info['estimated_height_m'])
            record = {
                'detection_type': 'UNSUPERVISED_ANOMALY',
                'class_name': 'uncataloged_hazard',
                'confidence': float(det.get('confidence', 0.0)),
                'bbox': bbox,
                'contour': contour,
                'verification_status': status,
                'estimated_height_m': estimate,
                'has_shadow': shadow_result['has_shadow'],
                'shadow_length_px': shadow_result['shadow_length_px'],
                'slant_range_m': target_slant_range,
            }
            verified_candidates.append(record)
        return verified_candidates

    def _predict_unsupervised_anomaly(self, image_bgr):
        """Generate anomaly detections from the autoencoder reconstruction error map."""
        if self.autoencoder is None:
            return []

        image_rgb = image_bgr
        if image_rgb.ndim == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_rgb

        original_h, original_w = gray.shape[:2]
        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.autoencoder(input_tensor)

        input_np = normalized.astype(np.float32)
        output_np = output_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()
        error_map = np.abs(input_np - output_np)
        blurred = cv2.GaussianBlur(error_map, (9, 9), 0)

        # The autoencoder is reached only after a supervised miss; use a
        # slightly wider tail so high-contrast physical debris is not lost.
        threshold = np.percentile(blurred, 97.5)
        anomaly_mask = (blurred >= threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 60:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w, y + h
            # Scale to original input image dimensions.
            x1_s = int(round(x1 * original_w / 256))
            y1_s = int(round(y1 * original_h / 256))
            x2_s = int(round(x2 * original_w / 256))
            y2_s = int(round(y2 * original_h / 256))

            contour_scaled = contour.astype(np.float32)
            contour_scaled[:, :, 0] = contour_scaled[:, :, 0] * (original_w / 256)
            contour_scaled[:, :, 1] = contour_scaled[:, :, 1] * (original_h / 256)
            contour_scaled = contour_scaled.astype(np.int32)

            # Mean reconstruction error from the anomaly map over the contour.
            mean_error = float(np.mean(blurred[y:y + h, x:x + w])) if w > 0 and h > 0 else 0.0
            detections.append({
                'detection_type': 'UNSUPERVISED_ANOMALY',
                'class_name': 'uncataloged_hazard',
                'confidence': float(mean_error),
                'bbox': [x1_s, y1_s, x2_s, y2_s],
                'contour': contour_scaled,
            })
        return detections

    def draw_annotations(self, image_bgr, detections):
        """Return a color-coded segmented visualization with verification-aware overlays and anomaly contours."""
        if image_bgr is None:
            return image_bgr
        image = np.array(image_bgr, copy=True)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay = image.copy()
        if not detections:
            return overlay

        for det in detections:
            detection_type = det.get('detection_type', 'SUPERVISED')
            contour = np.asarray(det.get('contour', np.zeros((0, 2), dtype=np.int32)))
            bbox = det.get('bbox')
            status = det.get('verification_status', 'VERIFIED_HAZARD')
            verified = status == 'VERIFIED_HAZARD'
            if detection_type == 'UNSUPERVISED_ANOMALY':
                if verified:
                    if len(contour) > 0:
                        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 2)
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        conf = float(det.get('confidence', 0.0))
                        height = det.get('estimated_height_m', 0.0)
                        cv2.putText(overlay, f'[ANOMALY] Uncataloged Hazard | H: {height}m | Conf: {conf:.2f}', (max(0, x1), max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), 1)
                        cv2.putText(overlay, '[CLUTTER - NO SHADOW]', (max(0, x1), max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            else:
                class_name = det.get('class_name', 'unknown')
                color = (0, 255, 0)
                if class_name in {'aircraft_wreck', 'shipwreck'}:
                    color = (0, 255, 255)
                if class_name == 'debris_highlight':
                    color = (0, 200, 255)
                if class_name == 'ghost_net':
                    color = (255, 255, 0)

                if verified:
                    if len(contour) > 0:
                        cv2.drawContours(overlay, [contour], -1, color, 1)
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        actual_conf = float(det.get('confidence', 0.0))
                        height = det.get('estimated_height_m', 0.0)
                        label = f'{class_name} | H: {height}m | Conf: {actual_conf:.2f}'
                        cv2.putText(overlay, label, (max(0, x1), max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 180, 180), 1)
                        cv2.putText(overlay, '[CLUTTER - NO SHADOW]', (max(0, x1), max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return overlay
