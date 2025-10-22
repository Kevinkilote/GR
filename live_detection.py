#!/usr/bin/env python3
"""CARLA manual control with optional YOLOv11 live detection overlay."""
from __future__ import annotations

import argparse
import logging
import math
import re
import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pygame

import manual_control_steeringwheel as base
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

from speed_limit_ocr import SpeedLimitOCR
from traffic_sign_recognition import DEFAULT_SIGN_LABELS, OTHER_SIGN_LABEL, RESNET_CLASS_NAMES


BOX_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    'traffic sign': (0, 255, 0),
    'traffic light': (255, 255, 0),
    'vehicle': (0, 170, 255),
    'car': (0, 170, 255),
    'speed-limit': (0, 220, 0),
    OTHER_SIGN_LABEL: (160, 160, 160),
}
DEFAULT_BOX_COLOR: Tuple[int, int, int] = (255, 255, 255)
SPEED_LIMIT_VALUES: Set[str] = {
    '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60',
    '65', '70', '75', '80', '85', '90', '95', '100', '110', '120', '130'
}
CLASS_CATEGORY_MAP: Dict[str, str] = {
    name.lower(): name.split('--', 1)[0] for name in RESNET_CLASS_NAMES
}
DEFAULT_CLASS_PRIORITY_WEIGHTS: Dict[str, float] = {
    'regulatory': 1.0,
    'warning': 0.7,
    'information': 0.4,
    'other': 0.4,
}
DEFAULT_LABEL_PRIORITY_WEIGHTS: Dict[str, float] = {
    name.lower(): DEFAULT_CLASS_PRIORITY_WEIGHTS.get(
        CLASS_CATEGORY_MAP.get(name.lower(), 'other'),
        DEFAULT_CLASS_PRIORITY_WEIGHTS['other'],
    )
    for name in RESNET_CLASS_NAMES
}
DEFAULT_LABEL_PRIORITY_WEIGHTS.update({
    'regulatory--yield--g1': 1.2,
    'regulatory--stop--g1': 1.3,
    'regulatory--maximum-speed-limit--g1': 1.0,
    'speed-limit': 1.0,
    'speed limit': 1.0,
    'yield': 1.2,
    'stop': 1.3,
    'traffic-light-red': 1.1,
    'traffic-light-green': 1.0,
    'traffic-light-yellow': 1.0,
})
DEFAULT_SPEED_OVER_WEIGHT = 1.2
DEFAULT_SPEED_TOLERANCE_KPH = 2.0
TRAFFIC_LIGHT_STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    'red': (255, 0, 0),
    'yellow': (255, 255, 0),
    'green': (0, 220, 0),
}


def parse_sign_labels_arg(raw: Optional[str]) -> Set[str]:
    if raw is None:
        return set(DEFAULT_SIGN_LABELS)
    normalized = raw.strip().lower()
    if normalized in {'', 'none', 'off'}:
        return set()
    labels = {label.strip().lower() for label in raw.split(',') if label.strip()}
    return labels


@dataclass(frozen=True)
class DetectionItem:
    bbox: Tuple[int, int, int, int]
    text: str
    color: Tuple[int, int, int]
    label: str
    confidence: float
    sign_class: Optional[str]
    category: str
    speed_limit: Optional[float]
    yolo_label: str


@dataclass(frozen=True)
class DetectionSnapshot:
    items: Tuple[DetectionItem, ...]
    timestamp: float


@dataclass(frozen=True)
class SignCacheEntry:
    bbox: Tuple[int, int, int, int]
    label: str
    base_class: Optional[str]
    confidence: float
    timestamp: float


class DetectionContext:
    """Lazy loads a YOLO model and throttles inference for live overlays."""

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
        min_interval: float = 0.1,
        resnet_path: Optional[str] = None,
        sign_labels: Optional[Set[str]] = None,
        display_ttl: float = 0.3,
        sign_cache_ttl: float = 0.75,
        sign_confidence_threshold: float = 0.6,
        skip_labels: Optional[Set[str]] = None,
        task_focus: str = 'none',
        show_overlays: bool = True,
    ) -> None:
        self.weights_path = weights_path
        self.confidence = conf
        self.iou = iou
        self.device = device
        self.min_interval = max(0.0, min_interval)
        self.display_ttl = max(0.05, display_ttl)
        self.sign_cache_ttl = max(0.1, sign_cache_ttl)
        self.sign_conf_threshold = max(0.0, sign_confidence_threshold)
        self.active = False
        self._model = None
        self._load_error: Optional[Exception] = None
        self._last_inference_time = 0.0
        self._last_snapshot: Optional[DetectionSnapshot] = None
        self._last_nonempty_snapshot: Optional[DetectionSnapshot] = None
        self._result_lock = threading.Lock()
        self._frame_queue: Optional[queue.Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._class_map: Optional[Dict[int, str]] = None
        self._resnet_path = resnet_path
        self._sign_labels = {label.lower() for label in (sign_labels or set())}
        self._skip_labels = {label.lower() for label in (skip_labels or set())}
        self._resnet_model = None
        self._resnet_transform = None
        self._torch_device: Optional[torch.device] = None
        self._sign_cache: List[SignCacheEntry] = []
        self._speed_limit_ocr = SpeedLimitOCR()
        self.class_priority_weights: Dict[str, float] = dict(DEFAULT_CLASS_PRIORITY_WEIGHTS)
        self.label_priority_weights: Dict[str, float] = dict(DEFAULT_LABEL_PRIORITY_WEIGHTS)
        self.speed_priority_boost = DEFAULT_SPEED_OVER_WEIGHT
        self.speed_tolerance_kph = DEFAULT_SPEED_TOLERANCE_KPH
        self.task_focus = task_focus.lower().strip()
        self.show_overlays = bool(show_overlays)
        self.min_priority_frames = 3

    def toggle(self) -> Tuple[bool, Optional[Exception]]:
        """Toggle detection on/off, attempting to load the model on-demand."""
        if self.active:
            self._stop_worker()
            self.active = False
            return True, None
        try:
            self._ensure_model()
            if self._sign_labels:
                self._ensure_sign_recognizer()
                self._sign_cache.clear()
            self._start_worker()
            self.active = True
            return True, None
        except Exception as exc:  # pylint: disable=broad-except
            self.active = False
            self._load_error = exc
            return False, exc

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError("ultralytics package is required for live detection") from exc
        self._model = YOLO(self.weights_path)
        names_attr = getattr(self._model, 'names', None)
        if isinstance(names_attr, dict):
            self._class_map = {int(idx): name for idx, name in names_attr.items()}
        elif isinstance(names_attr, Sequence):
            self._class_map = {idx: name for idx, name in enumerate(names_attr)}
        else:
            self._class_map = {}
        return self._model

    def submit_frame(self, frame_rgb: np.ndarray) -> None:
        if not self.active or self._frame_queue is None:
            return
        if not isinstance(frame_rgb, np.ndarray):
            return
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(frame_rgb)
        except queue.Full:
            logging.getLogger(__name__).debug('Detection frame queue is full; dropping frame')

    def get_latest_result(self) -> Optional[Tuple[DetectionItem, ...]]:
        with self._result_lock:
            snapshot = self._last_snapshot
            last_nonempty = self._last_nonempty_snapshot
        if snapshot is None and last_nonempty is None:
            return None
        now = time.time()
        if snapshot is not None:
            age = now - snapshot.timestamp
            if snapshot.items:
                if age <= self.display_ttl:
                    return snapshot.items
            else:
                if last_nonempty is not None:
                    nonempty_age = now - last_nonempty.timestamp
                    if nonempty_age <= self.display_ttl:
                        return last_nonempty.items
                if age <= self.display_ttl:
                    return tuple()
                return tuple()
        if last_nonempty is not None:
            nonempty_age = now - last_nonempty.timestamp
            if nonempty_age <= self.display_ttl:
                return last_nonempty.items
        return tuple()

    def shutdown(self) -> None:
        self._stop_worker()
        self.active = False
        self._sign_cache.clear()

    def _start_worker(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=1)
        with self._result_lock:
            self._last_snapshot = None
            self._last_nonempty_snapshot = None
        self._thread = threading.Thread(target=self._worker, name='yolo_detection_worker', daemon=True)
        self._thread.start()

    def _stop_worker(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._stop_event = None
        self._frame_queue = None
        with self._result_lock:
            self._last_snapshot = None
            self._last_nonempty_snapshot = None

    def _worker(self) -> None:
        logger = logging.getLogger(__name__)
        if self._model is None:
            # Model should already be loaded, but guard anyway.
            try:
                self._ensure_model()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error('Failed to load YOLO model: %s', exc)
                return
        if self._sign_labels:
            try:
                self._ensure_sign_recognizer()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error('Failed to load ResNet model: %s', exc)
                self._sign_labels = set()
        last_inference = 0.0
        assert self._frame_queue is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                frame_rgb = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            frame_rgb = np.ascontiguousarray(frame_rgb)
            now = time.time()
            if now - last_inference < self.min_interval:
                continue
            inference_kwargs = dict(conf=self.confidence, iou=self.iou, verbose=False)
            if self.device:
                inference_kwargs["device"] = self.device
            try:
                results = self._model(frame_rgb[:, :, ::-1], **inference_kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error('YOLO inference failed: %s', exc)
                with self._result_lock:
                    self._last_snapshot = None
                    self._last_nonempty_snapshot = None
                self._load_error = exc
                continue
            last_inference = now
            with self._result_lock:
                items = ()
                if results:
                    items = self._build_detection_items(results[0], frame_rgb, last_inference)
                snapshot = DetectionSnapshot(items=tuple(items), timestamp=last_inference)
                self._last_snapshot = snapshot
                if snapshot.items:
                    self._last_nonempty_snapshot = snapshot
                self._last_inference_time = last_inference

    def _ensure_sign_recognizer(self) -> None:
        if self._resnet_model is not None or not self._resnet_path:
            return
        if not os.path.exists(self._resnet_path):
            logging.getLogger(__name__).warning('ResNet weights not found at %s; disabling sign recognition', self._resnet_path)
            self._sign_labels = set()
            return
        if self._torch_device is None:
            if self.device:
                self._torch_device = torch.device(self.device)
            else:
                self._torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet_model = torchvision.models.resnet18(weights=None)
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = torch.nn.Linear(num_ftrs, len(RESNET_CLASS_NAMES))
        state_dict = torch.load(self._resnet_path, map_location=self._torch_device)
        resnet_model.load_state_dict(state_dict)
        self._resnet_model = resnet_model.to(self._torch_device)
        self._resnet_model.eval()
        self._resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_detection_items(self, result, frame_rgb: np.ndarray, timestamp: float) -> Tuple[DetectionItem, ...]:
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return tuple()
        class_map = self._class_map or {}
        frame_height, frame_width = frame_rgb.shape[:2]
        np_boxes = boxes.xyxy.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        items = []
        for bbox, score, cls_idx in zip(np_boxes, scores, classes):
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            x2 = max(0, min(frame_width - 1, x2))
            y2 = max(0, min(frame_height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            raw_label = class_map.get(int(cls_idx), f'class_{cls_idx}')
            label_lower = raw_label.lower()
            if label_lower in self._skip_labels:
                continue
            color = self._resolve_color(label_lower)
            display_label = raw_label
            display_confidence = float(score)
            sign_class: Optional[str] = None
            if label_lower in {'traffic light', 'traffic_light'}:
                traffic_result = self._classify_traffic_light(frame_rgb, (x1, y1, x2, y2))
                if traffic_result is not None:
                    state, state_conf = traffic_result
                    display_label = f"Traffic Light {state.capitalize()}"
                    display_confidence = (display_confidence + state_conf) / 2.0
                    sign_class = f"traffic-light-{state}"
                    color = TRAFFIC_LIGHT_STATE_COLORS.get(state, color)
            if label_lower in self._sign_labels and self._resnet_model is not None and self._resnet_transform is not None:
                classified = self._classify_sign(frame_rgb, (x1, y1, x2, y2), timestamp)
                if classified is not None:
                    sign_label, confidence, base_class = classified
                    display_label = self._format_sign_label(sign_label)
                    if sign_label == OTHER_SIGN_LABEL:
                        continue
                    elif sign_label == 'speed-limit' or sign_label.startswith('speed-limit-'):
                        color = self._resolve_color('speed-limit')
                    else:
                        color = self._resolve_color('traffic sign')
                    display_confidence = confidence
                    if base_class:
                        sign_class = base_class.lower()
            speed_limit_value = self._parse_speed_limit(display_label)
            category = self._categorize_sign(sign_class, display_label, raw_label)
            text = f"{display_label} ({display_confidence:.2f})"
            items.append(DetectionItem(
                bbox=(x1, y1, x2, y2),
                text=text,
                color=color,
                label=display_label,
                confidence=display_confidence,
                sign_class=sign_class,
                category=category,
                speed_limit=speed_limit_value,
                yolo_label=raw_label,
            ))
        return tuple(items)

    @staticmethod
    def _resolve_color(label_lower: str) -> Tuple[int, int, int]:
        if label_lower.startswith('speed-limit'):
            return BOX_COLOR_MAP.get('speed-limit', BOX_COLOR_MAP.get('traffic sign', DEFAULT_BOX_COLOR))
        return BOX_COLOR_MAP.get(label_lower, DEFAULT_BOX_COLOR)

    def _classify_sign(self, frame_rgb: np.ndarray, bbox: Tuple[int, int, int, int], timestamp: float) -> Optional[Tuple[str, float, Optional[str]]]:
        if self._resnet_model is None or self._resnet_transform is None or self._torch_device is None:
            return None
        cached = self._lookup_sign_cache(bbox, timestamp)
        if cached is not None:
            return cached.label, cached.confidence, cached.base_class
        x1, y1, x2, y2 = bbox
        crop_rgb = frame_rgb[y1:y2, x1:x2]
        if crop_rgb.size == 0:
            return None
        image = Image.fromarray(crop_rgb)
        input_tensor = self._resnet_transform(image).unsqueeze(0).to(self._torch_device)
        with torch.no_grad():
            output = self._resnet_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        base_class = RESNET_CLASS_NAMES[predicted_idx.item()]
        base_class_lower = base_class.lower()
        label = base_class
        confidence_value = float(confidence.item())

        ocr_result = self._speed_limit_ocr.infer(crop_rgb)
        if ocr_result is not None and ocr_result.score >= 0.45:
            label = f"speed-limit-{ocr_result.value}"
            confidence_value = float((confidence_value + ocr_result.score) / 2.0)
            base_class_lower = 'regulatory--maximum-speed-limit--g1'
        else:
            if 'maximum-speed-limit' in base_class_lower:
                extracted = self._extract_speed_value(base_class)
                if extracted is not None:
                    label = f"speed-limit-{extracted}"
                else:
                    label = 'speed-limit'
            elif base_class_lower.startswith('speed-limit'):
                extracted = self._extract_speed_value(base_class)
                if extracted is not None:
                    label = f"speed-limit-{extracted}"

        if confidence_value < self.sign_conf_threshold:
            return OTHER_SIGN_LABEL, confidence_value, base_class_lower
        self._update_sign_cache(bbox, label, base_class_lower, confidence_value, timestamp)
        return label, confidence_value, base_class_lower

    def _classify_traffic_light(self, frame_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[str, float]]:
        x1, y1, x2, y2 = bbox
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None
        resized = cv2.resize(crop, (60, 120), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        red_mask1 = cv2.inRange(hsv, (0, 90, 120), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 90, 120), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        yellow_mask = cv2.inRange(hsv, (15, 90, 120), (35, 255, 255))
        green_mask = cv2.inRange(hsv, (40, 90, 100), (90, 255, 255))
        height = hsv.shape[0]
        thirds = [
            (0, int(height * 0.33)),
            (int(height * 0.33), int(height * 0.66)),
            (int(height * 0.66), height),
        ]

        def region_score(mask: np.ndarray, region: Tuple[int, int]) -> float:
            start, end = region
            roi = mask[start:end, :]
            if roi.size == 0:
                return 0.0
            return float(np.sum(roi)) / (roi.size * 255.0)

        red_score = region_score(red_mask, thirds[0]) * 1.3 + float(np.sum(red_mask)) / (red_mask.size * 255.0) * 0.4
        yellow_score = region_score(yellow_mask, thirds[1]) * 1.3 + float(np.sum(yellow_mask)) / (yellow_mask.size * 255.0) * 0.4
        green_score = region_score(green_mask, thirds[2]) * 1.3 + float(np.sum(green_mask)) / (green_mask.size * 255.0) * 0.4
        scores = {
            'red': red_score,
            'yellow': yellow_score,
            'green': green_score,
        }
        best_state, best_score = max(scores.items(), key=lambda item: item[1])
        total = sum(scores.values())
        if total <= 0 or best_score < 0.05:
            return None
        confidence = best_score / (total + 1e-6)
        return best_state, confidence

    def _lookup_sign_cache(self, bbox: Tuple[int, int, int, int], timestamp: float) -> Optional[SignCacheEntry]:
        if not self._sign_cache:
            return None
        retained: List[SignCacheEntry] = []
        best_entry: Optional[SignCacheEntry] = None
        best_iou = 0.0
        for entry in self._sign_cache:
            if timestamp - entry.timestamp > self.sign_cache_ttl:
                continue
            retained.append(entry)
            iou = self._compute_iou(bbox, entry.bbox)
            if iou > 0.55 and iou > best_iou:
                best_entry = entry
                best_iou = iou
        self._sign_cache = retained
        if best_entry and best_entry.label == OTHER_SIGN_LABEL:
            return None
        return best_entry

    def _update_sign_cache(self, bbox: Tuple[int, int, int, int], label: str, base_class: Optional[str], confidence: float, timestamp: float) -> None:
        entry = SignCacheEntry(bbox=bbox, label=label, base_class=base_class.lower() if base_class else None, confidence=confidence, timestamp=timestamp)
        self._sign_cache.append(entry)
        if len(self._sign_cache) > 32:
            self._sign_cache = self._sign_cache[-32:]

    @staticmethod
    def _compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def _format_sign_label(label: str) -> str:
        if label == OTHER_SIGN_LABEL:
            return 'Other Sign'
        if label == 'speed-limit':
            return 'Speed Limit'
        if label.startswith('speed-limit-'):
            suffix = label.split('-', 2)[2]
            return f"Speed Limit {suffix}"
        parts = label.replace('--', ' ').replace('-', ' ').replace('_', ' ').split()
        return ' '.join(part.capitalize() for part in parts)

    @staticmethod
    def _extract_speed_value(label: str) -> Optional[str]:
        label_lower = label.lower()
        specific_match = re.search(r'(?:speed[-_]?limit[-_]?)(\d{1,3})', label_lower)
        if specific_match:
            return specific_match.group(1)
        max_match = re.search(r'(?:maximum[-_]?speed[-_]?limit[-_]?)(\d{1,3})', label_lower)
        if max_match:
            return max_match.group(1)
        for match in re.finditer(r'\d{1,3}', label_lower):
            digits = match.group()
            if digits in SPEED_LIMIT_VALUES:
                return digits
            if len(digits) >= 2 and digits.lstrip('0') in SPEED_LIMIT_VALUES:
                return digits.lstrip('0')
            start = match.start()
            if start > 0 and label_lower[start - 1] == 'g':
                continue
            if len(digits) >= 2:
                return digits
        return None

    @staticmethod
    def _categorize_sign(sign_class: Optional[str], display_label: str, fallback_label: str) -> str:
        if sign_class:
            lowered = sign_class.lower()
            if lowered.startswith('traffic-light'):
                return 'regulatory'
            category = CLASS_CATEGORY_MAP.get(lowered)
            if category:
                return category
        for candidate in (display_label.lower(), fallback_label.lower()):
            if candidate.startswith('regulatory') or 'speed limit' in candidate or 'yield' in candidate or 'stop' in candidate:
                return 'regulatory'
            if candidate.startswith('warning') or 'warning' in candidate:
                return 'warning'
            if candidate.startswith('information') or 'information' in candidate or 'parking' in candidate:
                return 'information'
        return 'other'

    def _parse_speed_limit(self, display_label: str) -> Optional[float]:
        digits = self._extract_speed_value(display_label)
        if digits is None:
            return None
        if digits not in SPEED_LIMIT_VALUES:
            return None
        try:
            value = float(digits)
        except ValueError:
            return None
        return value



class LiveDetectionCameraManager(base.CameraManager):
    """Camera manager that overlays YOLO detections when enabled."""

    BG_COLOR = (20, 20, 20)
    LABEL_CACHE_MAX = 64

    def __init__(
        self,
        parent_actor,
        hud,
        detection: DetectionContext,
        frame_interval: Optional[float] = None,
        show_overlays: bool = True,
    ):
        super().__init__(parent_actor, hud)
        self._detection = detection
        self._frame_interval = frame_interval if frame_interval and frame_interval > 0 else None
        self._label_font = pygame.font.Font(pygame.font.get_default_font(), 18)
        self._priority_font = pygame.font.Font(pygame.font.get_default_font(), 24)
        self._surface_size: Optional[Tuple[int, int]] = None
        self._label_cache: Dict[Tuple[str, Tuple[int, int, int]], pygame.Surface] = {}
        self._show_overlays = show_overlays
        self._priority_label: Optional[str] = None
        self._priority_score: float = 0.0
        self._priority_timestamp: float = 0.0
        self._class_weights = {key.lower(): value for key, value in detection.class_priority_weights.items()}
        self._label_weights = {key.lower(): value for key, value in detection.label_priority_weights.items()}
        self._speed_over_weight = detection.speed_priority_boost
        self._speed_tolerance_kph = detection.speed_tolerance_kph
        self._task_focus = detection.task_focus
        self._current_speed = 0.0
        self._history_window = 10
        self._recent_detections: deque[Sequence[DetectionItem]] = deque(maxlen=self._history_window)
        self._min_priority_occurrences = max(1, getattr(detection, 'min_priority_frames', 3))
        self._priority_category: Optional[str] = None
        self._priority_start_time: float = 0.0
        self._cooldown_until: float = 0.0
        self._priority_source_label: Optional[str] = None
        self._priority_sign_class: Optional[str] = None
        interior_view = base.carla.Transform(
            base.carla.Location(x=0.5, y=0.0, z=1.2),
            base.carla.Rotation(pitch=0.0),
        )
        hood_view = base.carla.Transform(
            base.carla.Location(x=1.6, z=1.7),
            base.carla.Rotation(pitch=0.0),
        )
        chase_view = base.carla.Transform(
            base.carla.Location(x=-5.5, z=2.8),
            base.carla.Rotation(pitch=-15),
        )
        existing_transforms = list(getattr(self, '_camera_transforms', []))
        desired_order = [interior_view, hood_view, chase_view]
        merged: List[base.carla.Transform] = []
        for transform in desired_order + existing_transforms:
            if not any(self._transforms_equal(transform, current) for current in merged):
                merged.append(transform)
        self._camera_transforms = merged
        self.transform_index = 0

    def set_sensor(self, index, notify=True):  # noqa: D401 (interface inherited)
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            blueprint = self.sensors[index][-1]
            if self._frame_interval and blueprint.has_attribute('sensor_tick'):
                blueprint.set_attribute('sensor_tick', f'{self._frame_interval:.6f}')
            self.sensor = self._parent.get_world().spawn_actor(
                blueprint,
                self._camera_transforms[self.transform_index],
                attach_to=self._parent,
            )
            weak_self = base.weakref.ref(self)
            self.sensor.listen(lambda image: LiveDetectionCameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.index is None or self.index >= len(self.sensors):
            return
        sensor_id = self.sensors[self.index][0]
        if sensor_id.startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data).astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img = np.zeros((self.hud.dim[0], self.hud.dim[1], 3))
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
            return

        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        rgb_array = array[:, :, ::-1]
        self._detection.submit_frame(rgb_array)
        surface = self._ensure_surface(image.width, image.height)
        pygame.surfarray.blit_array(surface, np.ascontiguousarray(rgb_array.swapaxes(0, 1)))
        detection_result = self._detection.get_latest_result()
        if detection_result is not None:
            self._draw_detections(surface, detection_result)
        self.surface = surface

    def _ensure_surface(self, width: int, height: int) -> pygame.Surface:
        size = (width, height)
        if self.surface is None or self._surface_size != size:
            self.surface = pygame.Surface(size).convert()
            self._surface_size = size
        return self.surface

    def _draw_detections(self, surface: pygame.Surface, detections: Sequence[DetectionItem]) -> None:
        width, height = surface.get_width(), surface.get_height()
        detection_list = list(detections)
        if self._show_overlays:
            for detection in detection_list:
                x1, y1, x2, y2 = detection.bbox
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                self._draw_label(surface, x1, y1, detection.text, detection.color)
        self._update_priority_panel(detection_list)
        self._draw_priority_panel(surface)

    def _draw_label(self, surface: pygame.Surface, x: int, y: int, label: str, color: Tuple[int, int, int]) -> None:
        label_surface = self._get_label_surface(label, color)
        position_y = max(0, y - label_surface.get_height())
        surface.blit(label_surface, (x, position_y))

    def _get_label_surface(self, label: str, color: Tuple[int, int, int]) -> pygame.Surface:
        cache_key = (label, color)
        cached = self._label_cache.get(cache_key)
        if cached is None:
            text_surface = self._label_font.render(label, True, color)
            label_surface = pygame.Surface((text_surface.get_width() + 6, text_surface.get_height() + 6), pygame.SRCALPHA)
            label_surface.fill((*self.BG_COLOR, 160))
            label_surface.blit(text_surface, (3, 3))
            if len(self._label_cache) >= self.LABEL_CACHE_MAX:
                self._label_cache.clear()
            self._label_cache[cache_key] = label_surface
            cached = label_surface
        return cached.copy()

    def _compute_priority_score(self, detection: DetectionItem, current_speed: float) -> float:
        class_key = detection.category.lower() if detection.category else 'other'
        class_weight = self._class_weights.get(class_key, self._class_weights.get('other', 0.4))
        label_weight = None
        if detection.sign_class:
            label_weight = self._label_weights.get(detection.sign_class, None)
        if label_weight is None:
            display = detection.label.lower()
            for key, weight in self._label_weights.items():
                if display.startswith(key):
                    label_weight = weight
                    break
        if label_weight is None:
            label_weight = self._label_weights.get(detection.yolo_label.lower())
        effective_weight = label_weight if label_weight is not None else class_weight
        if detection.speed_limit is not None and detection.speed_limit > 0:
            if current_speed - detection.speed_limit > self._speed_tolerance_kph:
                effective_weight = max(effective_weight, self._speed_over_weight)
        tta_factor = self._compute_tta_factor(detection, current_speed)
        confidence = self._compute_confidence_factor(detection.confidence)
        task_boost = self._compute_task_boost(detection)
        score = (
            0.50 * effective_weight
            + 0.25 * tta_factor
            + 0.15 * confidence
            + 0.10 * task_boost
        )
        return score

    def _estimate_distance(self, detection: DetectionItem) -> float:
        x1, y1, x2, y2 = detection.bbox
        bbox_height = max(1, y2 - y1)
        frame_height = getattr(self, '_surface_size', None)
        if frame_height and isinstance(frame_height, tuple):
            total_height = frame_height[1]
        else:
            total_height = self.hud.dim[1] if hasattr(self.hud, 'dim') else 720
        ratio = min(0.95, bbox_height / max(1, total_height))
        ratio = max(ratio, 0.02)
        approximate_distance = 8.0 / ratio
        return approximate_distance

    def _compute_tta_factor(self, detection: DetectionItem, current_speed: float) -> float:
        distance = self._estimate_distance(detection)
        speed_mps = max(current_speed / 3.6, 0.1)
        tta = distance / speed_mps
        if tta <= 2.0:
            return 1.0
        if tta <= 6.0:
            return 1.0 - ((tta - 2.0) / 4.0) * (1.0 - 0.4)
        return 0.2

    @staticmethod
    def _compute_confidence_factor(confidence: float) -> float:
        confidence = max(0.0, min(1.0, confidence))
        if confidence <= 0.5:
            return 0.6
        if confidence >= 0.9:
            return 1.0
        return 0.6 + ((confidence - 0.5) / 0.4) * (1.0 - 0.6)

    def _compute_task_boost(self, detection: DetectionItem) -> float:
        focus = self._task_focus
        label = detection.label.lower()
        sign_class = (detection.sign_class or '').lower() if detection.sign_class else ''
        category = detection.category.lower() if detection.category else 'other'
        if focus == 'none':
            return 0.0
        if focus == 'parking':
            if 'parking' in label:
                return 0.15
            if 'no parking' in label or 'no-parking' in label:
                return 0.10
        elif focus in {'speed', 'navigation'}:
            if 'speed limit' in label or (sign_class and 'speed-limit' in sign_class):
                return 0.15
            if any(keyword in label for keyword in ('turn', 'go straight', 'keep right', 'keep left')):
                return 0.12
        elif focus == 'safety':
            if 'pedestrian' in label or 'children' in label or 'school' in label:
                return 0.15
        return 0.0

    @staticmethod
    def _simplify_hierarchical_label(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        lowered = value.strip().lower()
        if not lowered:
            return None
        if '--' in lowered:
            parts = lowered.split('--')
            parts = parts[1:]  # drop prefix category
        else:
            parts = lowered.split('-')
        if parts and re.fullmatch(r'g\d+[a-z]?$', parts[-1]):
            parts = parts[:-1]
        if not parts:
            return None
        text = ' '.join(parts)
        text = text.replace('-', ' ')
        text = ' '.join(word.capitalize() for word in text.split())
        return text or None

    @staticmethod
    def _clean_display_label(label: str) -> str:
        if not label:
            return label
        text = label.strip()
        if not text:
            return text
        text = re.sub(r'\b(Regulatory|Warning|Information)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bG\d+[A-Z]?\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return label
        tokens = []
        for token in text.split():
            if token.isalpha():
                tokens.append(token.capitalize())
            else:
                tokens.append(token)
        return ' '.join(tokens)

    def _format_priority_label(self, display_label: str, raw_label: str, sign_class: Optional[str]) -> str:
        if sign_class and 'speed-limit' in sign_class:
            cleaned_speed = self._clean_display_label(display_label)
            if re.search(r'\d', cleaned_speed):
                return cleaned_speed
        candidate = self._simplify_hierarchical_label(sign_class)
        if candidate:
            return candidate
        candidate = self._simplify_hierarchical_label(raw_label)
        if candidate:
            return candidate
        cleaned_display = self._clean_display_label(display_label)
        if cleaned_display:
            return cleaned_display
        cleaned_raw = self._clean_display_label(raw_label)
        if cleaned_raw:
            return cleaned_raw
        return display_label

    def _update_priority_panel(self, detections: Sequence[DetectionItem]) -> None:
        now = time.time()
        try:
            velocity = self._parent.get_velocity()
            current_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
        except Exception:  # pylint: disable=broad-except
            current_speed = 0.0
        self._current_speed = current_speed
        self._recent_detections.append(tuple(detections))
        if not self._recent_detections:
            self._priority_label = None
            self._priority_score = 0.0
            self._priority_timestamp = now
            return

        aggregated: Dict[str, Dict[str, object]] = defaultdict(lambda: {
            'score': 0.0,
            'count': 0,
            'last': None,
        })
        for frame_items in self._recent_detections:
            for detection in frame_items:
                entry = aggregated[detection.label]
                entry['score'] = float(entry['score']) + self._compute_priority_score(detection, current_speed)
                entry['count'] = int(entry['count']) + 1
                entry['last'] = detection

        best: Optional[Tuple[str, float, DetectionItem]] = None
        for label, data in aggregated.items():
            count = int(data['count']) if data['count'] is not None else 0
            if count < self._min_priority_occurrences:
                continue
            avg_score = float(data['score']) / max(count, 1)
            detection = data['last']
            if detection is None:
                continue
            if best is None or avg_score > best[1]:
                best = (label, avg_score, detection)

        current_entry = None
        current_score = 0.0
        current_detection: Optional[DetectionItem] = None
        if self._priority_source_label:
            entry = aggregated.get(self._priority_source_label)
            if entry and entry['last'] is not None:
                current_entry = entry
                current_detection = entry['last']
                current_score = float(entry['score']) / max(int(entry['count']), 1)

        if best is None:
            if self._priority_label is not None:
                self._cooldown_until = now + 0.7
            self._priority_label = None
            self._priority_source_label = None
            self._priority_score = 0.0
            self._priority_timestamp = now
            self._priority_category = None
            self._priority_sign_class = None
            return

        label, score, detection = best
        candidate_priority = self._format_priority_label(detection.label, detection.yolo_label, detection.sign_class)
        candidate_category = detection.category.lower() if detection.category else 'other'
        candidate_class_weight = self._class_weights.get(candidate_category, self._class_weights.get('other', 0.4))

        if current_detection is not None and self._priority_label is not None:
            current_category = self._priority_category or candidate_category
            current_class_weight = self._class_weights.get(current_category, self._class_weights.get('other', 0.4))
            dwell_elapsed = now - self._priority_start_time
            # dwell requirement: keep current for at least 1.0s if still visible
            if dwell_elapsed < 1.0:
                self._priority_label = self._format_priority_label(
                    current_detection.label,
                    current_detection.yolo_label,
                    current_detection.sign_class,
                )
                self._priority_score = current_score
                self._priority_timestamp = now
                return
            # regulatory lock
            if (
                current_category == 'regulatory'
                and candidate_category != 'regulatory'
                and dwell_elapsed < 3.0
                and score < current_score * 1.2
            ):
                self._priority_label = self._format_priority_label(
                    current_detection.label,
                    current_detection.yolo_label,
                    current_detection.sign_class,
                )
                self._priority_score = current_score
                self._priority_timestamp = now
                return
            # cooldown against lower priority
            if (
                now < self._cooldown_until
                and candidate_class_weight < current_class_weight
                and score < current_score * 1.2
            ):
                self._priority_label = self._format_priority_label(
                    current_detection.label,
                    current_detection.yolo_label,
                    current_detection.sign_class,
                )
                self._priority_score = current_score
                self._priority_timestamp = now
                return

        self._priority_label = candidate_priority
        self._priority_source_label = detection.label
        self._priority_score = score
        self._priority_timestamp = now
        self._priority_start_time = now
        self._priority_category = candidate_category
        self._priority_sign_class = detection.sign_class

    def _draw_priority_panel(self, surface: pygame.Surface) -> None:
        if not self._priority_label:
            return
        display_text = self._priority_label
        text_surface = self._priority_font.render(display_text, True, (255, 255, 255))
        padding = 12
        width = text_surface.get_width() + padding * 2
        height = text_surface.get_height() + padding * 2
        panel_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        panel_surface.fill((*self.BG_COLOR, 170))
        panel_surface.blit(text_surface, (padding, padding))
        x = max(0, (surface.get_width() - width) // 2)
        y = max(0, surface.get_height() - height - 20)
        surface.blit(panel_surface, (x, y))

    @staticmethod
    def _transforms_equal(a: 'base.carla.Transform', b: 'base.carla.Transform', tol: float = 1e-3) -> bool:
        loc_a, loc_b = a.location, b.location
        rot_a, rot_b = a.rotation, b.rotation
        return (
            abs(loc_a.x - loc_b.x) <= tol
            and abs(loc_a.y - loc_b.y) <= tol
            and abs(loc_a.z - loc_b.z) <= tol
            and abs(rot_a.pitch - rot_b.pitch) <= tol
            and abs(rot_a.roll - rot_b.roll) <= tol
            and abs(rot_a.yaw - rot_b.yaw) <= tol
        )


class LiveDetectionWorld(base.World):
    """World wrapper that swaps in a detection-aware camera manager."""

    def __init__(self, carla_world, hud, actor_filter, detection: DetectionContext, sim_fps: Optional[float]):
        self._detection = detection
        self._sim_fps = sim_fps if sim_fps and sim_fps > 0 else None
        self._original_settings = carla_world.get_settings()
        self._sync_mode_enabled = False
        if self._sim_fps:
            self._enable_sync_mode(carla_world, self._sim_fps)
        super().__init__(carla_world, hud, actor_filter)

    def restart(self):
        previous_manager = getattr(self, 'camera_manager', None)
        prev_transform_index = previous_manager.transform_index if previous_manager and previous_manager.transform_index is not None else 0
        prev_sensor_index = previous_manager.index if previous_manager and previous_manager.index is not None else 0
        if previous_manager is not None:
            previous_manager.transform_index = min(prev_transform_index, 1)
        super().restart()
        self._attach_detection_camera(prev_transform_index, prev_sensor_index)

    def _attach_detection_camera(self, transform_index_override: Optional[int] = None, sensor_index_override: Optional[int] = None) -> None:
        previous_manager = getattr(self, 'camera_manager', None)
        transform_index = transform_index_override if transform_index_override is not None else 0
        sensor_index = sensor_index_override if sensor_index_override is not None else 0
        if previous_manager is not None:
            if transform_index_override is None and previous_manager.transform_index is not None:
                transform_index = previous_manager.transform_index
            if sensor_index_override is None and previous_manager.index is not None:
                sensor_index = previous_manager.index
            if previous_manager.sensor is not None:
                previous_manager.sensor.stop()
                previous_manager.sensor.destroy()
        frame_interval = (1.0 / self._sim_fps) if self._sim_fps else None
        show_overlays = getattr(self._detection, 'show_overlays', True)
        self.camera_manager = LiveDetectionCameraManager(
            self.player,
            self.hud,
            self._detection,
            frame_interval,
            show_overlays=show_overlays,
        )
        max_transform = max(1, len(self.camera_manager._camera_transforms))
        self.camera_manager.transform_index = transform_index % max_transform
        self.camera_manager.set_sensor(sensor_index, notify=False)

    def _enable_sync_mode(self, carla_world, sim_fps: float) -> None:
        settings = carla_world.get_settings()
        target_delta = 1.0 / sim_fps
        current_delta = settings.fixed_delta_seconds or 0.0
        needs_update = (
            not settings.synchronous_mode
            or not math.isclose(current_delta, target_delta, rel_tol=1e-3)
        )
        if needs_update:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = target_delta
            carla_world.apply_settings(settings)
        self._sync_mode_enabled = True

    def destroy(self):
        try:
            super().destroy()
        finally:
            if self._sync_mode_enabled:
                self.world.apply_settings(self._original_settings)
                self._sync_mode_enabled = False

    def tick(self, clock):
        if self._sync_mode_enabled:
            self.world.tick()
        super().tick(clock)


class LiveDualControl(base.DualControl):
    """Extends DualControl to toggle live detection from input events."""

    def __init__(self, world, start_in_autopilot, detection: DetectionContext, detect_button: Optional[int]):
        self._detection = detection
        self._detect_button = detect_button if detect_button is None or detect_button >= 0 else None
        super().__init__(world, start_in_autopilot)

    def parse_events(self, world, clock):  # noqa: D401
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.JOYBUTTONDOWN:
                if self._detect_button is not None and event.button == self._detect_button:
                    self._handle_detection_toggle(world)
                    continue
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                if event.key == pygame.K_BACKSPACE:
                    world.restart()
                elif event.key == pygame.K_F1:
                    world.hud.toggle_info()
                elif event.key == pygame.K_h or (event.key == pygame.K_SLASH and pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == pygame.K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == pygame.K_c:
                    world.next_weather()
                elif event.key == pygame.K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > pygame.K_0 and event.key <= pygame.K_9:
                    world.camera_manager.set_sensor(event.key - 1 - pygame.K_0)
                elif event.key == pygame.K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == pygame.K_l:
                    self._handle_detection_toggle(world)
                if isinstance(self._control, base.carla.VehicleControl):
                    if event.key == pygame.K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == pygame.K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('Manual Transmission' if self._control.manual_gear_shift else 'Automatic Transmission')
                    elif self._control.manual_gear_shift and event.key == pygame.K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == pygame.K_PERIOD:
                        self._control.gear += 1
                    elif event.key == pygame.K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, base.carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, base.carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _handle_detection_toggle(self, world):
        success, error = self._detection.toggle()
        if success and self._detection.active:
            world.hud.notification('Live detection ON')
        elif success:
            world.hud.notification('Live detection OFF')
        else:
            world.hud.notification(f'Live detection failed: {error}')


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    detection_context = DetectionContext(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        min_interval=args.detection_interval,
        resnet_path=args.resnet,
        sign_labels=args.sign_labels,
        display_ttl=args.display_ttl,
        sign_cache_ttl=args.sign_cache_ttl,
        sign_confidence_threshold=args.sign_conf_threshold,
        skip_labels=args.skip_labels,
        task_focus=args.task_focus,
        show_overlays=args.show_overlays,
    )
    detection_context.min_priority_frames = args.min_priority_frames
    try:
        client = base.carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        hud = base.HUD(args.width, args.height)
        world = LiveDetectionWorld(client.get_world(), hud, args.filter, detection_context, args.sim_fps)
        controller = LiveDualControl(world, args.autopilot, detection_context, args.detect_button)
        clock = pygame.time.Clock()
        target_fps = int(round(args.sim_fps)) if args.sim_fps > 0 else 0
        while True:
            if target_fps > 0:
                clock.tick_busy_loop(target_fps)
            else:
                clock.tick_busy_loop(0)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
    finally:
        if world is not None:
            world.destroy()
        detection_context.shutdown()
        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CARLA manual control with optional YOLO live detection')
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    parser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    parser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    parser.add_argument('--weights', default='best.pt', help='path to YOLOv11 weights (default: best.pt)')
    parser.add_argument('--resnet', default='best_traffic_sign_classifier_advanced.pth', help='path to traffic sign ResNet weights (default: best_traffic_sign_classifier_advanced.pth; use "none" to disable)')
    parser.add_argument('--sign-labels', default=None, help='comma-separated YOLO class names to refine with ResNet (default uses known traffic sign labels)')
    parser.add_argument('--conf', default=0.4, type=float, help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--iou', default=0.45, type=float, help='YOLO IoU threshold (default: 0.45)')
    parser.add_argument('--device', default='cuda:0', help='Torch device for YOLO (default: cuda:0)')
    parser.add_argument('--detection-interval', default=0.05, type=float, help='minimum seconds between YOLO inferences (default: 0.05)')
    parser.add_argument('--display-ttl', default=0.3, type=float, help='seconds to keep last detections on screen (default: 0.3)')
    parser.add_argument('--sign-cache-ttl', default=0.75, type=float, help='seconds to reuse cached ResNet classifications (default: 0.75)')
    parser.add_argument('--sign-conf-threshold', default=0.65, type=float, help='minimum ResNet confidence before accepting a sign class (default: 0.6)')
    parser.add_argument('--skip-labels', default='', help='comma-separated YOLO class names to ignore (e.g., "car,truck")')
    parser.add_argument('--task-focus', default='none', choices=['none', 'parking', 'speed', 'navigation', 'safety'], help='task context to boost relevant signs')
    parser.add_argument('--hide-boxes', action='store_true', help='hide individual detection overlays and show only priority summary')
    parser.add_argument('--sim-fps', default=30.0, type=float, help='target simulation FPS for 1:1 playback (set to 0 to disable throttling)')
    parser.add_argument('--detect-button', default=None, type=int, help='joystick button index to toggle detection (omit to use keyboard only; set -1 to disable)')
    parser.add_argument('--debug', action='store_true', help='print debug information')
    parser.add_argument('--min-priority-frames', default=4, type=int, help='minimum frames a sign must persist before it can drive priority text (default: 3)')

    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    if args.detect_button is not None and args.detect_button < 0:
        args.detect_button = None
    if args.resnet and args.resnet.strip().lower() == 'none':
        args.resnet = None
    args.sim_fps = max(0.0, float(args.sim_fps))
    args.sign_labels = parse_sign_labels_arg(args.sign_labels)
    args.display_ttl = max(0.05, args.display_ttl)
    args.sign_cache_ttl = max(0.1, args.sign_cache_ttl)
    args.sign_conf_threshold = max(0.0, args.sign_conf_threshold)
    args.task_focus = args.task_focus.lower().strip()
    args.skip_labels = {label.strip().lower() for label in args.skip_labels.split(',') if label.strip()} if args.skip_labels else set()
    args.show_overlays = not args.hide_boxes
    args.min_priority_frames = max(1, int(args.min_priority_frames))
    return args


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
