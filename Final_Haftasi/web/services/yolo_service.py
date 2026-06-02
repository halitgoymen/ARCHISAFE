"""YOLO detection service — wraps the existing trained models."""
import os
import sys
import threading
import numpy as np

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from ultralytics import YOLO
            from flask import current_app

            paths = [
                current_app.config.get('YOLO_MODEL_V6', ''),
                current_app.config.get('YOLO_MODEL_V5', ''),
                current_app.config.get('YOLO_MODEL_V4', ''),
            ]
            for p in paths:
                if p and os.path.exists(p):
                    _model = YOLO(p)
                    return _model

            # Fallback: download yolo12n
            _model = YOLO('yolo12n.pt')
            return _model
        except Exception as e:
            print(f'[YOLO] Model yüklenemedi: {e}')
            return None


CLASS_NAMES = [
    'Ear-protection', 'Fall Detection', 'gloves', 'hardhat',
    'mask', 'no_arm_sleeve', 'person', 'shoes',
    'ungloves', 'unhardhat', 'unmask', 'unvest', 'vest'
]

VIOLATION_CLASSES = {'ungloves', 'unhardhat', 'unmask', 'unvest', 'no_arm_sleeve'}
SAFE_PPE_CLASSES = {'gloves', 'hardhat', 'mask', 'vest', 'shoes', 'Ear-protection'}
FALL_CLASS = 'Fall Detection'

PPE_DISPLAY_NAMES = {
    'hardhat': 'Baret',
    'vest': 'Yelek',
    'mask': 'Maske',
    'gloves': 'Eldiven',
    'shoes': 'Bota',
    'Ear-protection': 'Kulak Koruyucu',
    'unhardhat': 'Baretsiz',
    'unvest': 'Yeleksiz',
    'unmask': 'Maskesiz',
    'ungloves': 'Eldivensiz',
    'no_arm_sleeve': 'Kolsuz',
    'Fall Detection': 'Düşme',
    'person': 'Kişi',
}

COLOR_MAP = {
    'person': (0, 255, 255),
    'hardhat': (0, 255, 0),
    'vest': (0, 255, 0),
    'mask': (0, 255, 0),
    'gloves': (0, 255, 0),
    'shoes': (0, 255, 0),
    'Ear-protection': (0, 255, 0),
    'unhardhat': (0, 0, 255),
    'unvest': (0, 0, 255),
    'unmask': (0, 0, 255),
    'ungloves': (0, 0, 255),
    'no_arm_sleeve': (0, 165, 255),
    'Fall Detection': (0, 0, 255),
}


def detect_frame(frame, conf=0.35):
    """Run YOLO detection on a single frame.
    Returns list of dicts: {class_name, confidence, bbox, color}
    """
    model = _get_model()
    if model is None:
        return []

    try:
        results = model(frame, conf=conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = results.names.get(cls_idx, CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else 'unknown')
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                'class_name': cls_name,
                'confidence': conf_val,
                'bbox': (x1, y1, x2, y2),
                'color': COLOR_MAP.get(cls_name, (128, 128, 128)),
                'is_violation': cls_name in VIOLATION_CLASSES,
                'is_fall': cls_name == FALL_CLASS,
            })
        return detections
    except Exception as e:
        print(f'[YOLO] Detection error: {e}')
        return []


def annotate_frame(frame, detections):
    """Draw bounding boxes and labels on frame."""
    import cv2
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = det['color']
        label = f"{PPE_DISPLAY_NAMES.get(det['class_name'], det['class_name'])} {det['confidence']:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame


def extract_ppe_status(detections, required_ppe):
    """Given detections and required PPE list, return detected/missing."""
    detected = {d['class_name'] for d in detections if d['class_name'] in SAFE_PPE_CLASSES}
    required_set = set(required_ppe)
    missing = list(required_set - detected)
    present = list(detected & required_set)
    violations = [d['class_name'] for d in detections if d['is_violation']]
    return {
        'detected': list(detected),
        'missing': missing,
        'present': present,
        'violations': violations,
        'compliant': len(missing) == 0 and len(violations) == 0,
    }
