#!/usr/bin/env python3
"""Offline CARLA post-analysis: run YOLO + ResNet over a recorded video."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from traffic_sign_recognition import (
    DEFAULT_SIGN_LABELS,
    RESNET_CLASS_NAMES,
    ResNetBundle,
    load_resnet,
    normalise_sign_labels,
)


@dataclass(frozen=True)
class DetectionItem:
    bbox: Tuple[int, int, int, int]
    text: str
    color: Tuple[int, int, int]


@dataclass(frozen=True)
class DetectionSnapshot:
    items: Tuple[DetectionItem, ...]
    timestamp: float


@dataclass(frozen=True)
class SignCacheEntry:
    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float
    timestamp: float


BOX_COLOR_MAP = {
    'traffic sign': (0, 255, 0),      # green
    'traffic light': (0, 255, 255),   # yellow
    'vehicle': (0, 170, 255),         # orange/blue-ish
    'car': (0, 170, 255),
}
DEFAULT_BOX_COLOR = (255, 255, 255)


class VideoDetectionPipeline:
    """Applies YOLO + ResNet detections to individual video frames."""

    def __init__(
        self,
        yolo_weights: str,
        resnet_weights: Optional[str],
        *,
        conf: float,
        iou: float,
        device: Optional[str],
        sign_labels: Iterable[str],
        detection_interval: float,
        display_ttl: float,
        sign_cache_ttl: float,
    ) -> None:
        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
        self._yolo = YOLO(yolo_weights)
        self._conf = conf
        self._iou = iou
        self._device_override = device
        self._names = self._normalise_yolo_names(self._yolo.names)

        self._sign_labels = {label.lower() for label in sign_labels}
        self._resnet_bundle: Optional[ResNetBundle] = None
        if self._sign_labels and resnet_weights:
            self._resnet_bundle = load_resnet(resnet_weights, device_hint=device)
        elif self._sign_labels:
            logging.warning('Sign labels supplied but ResNet weights missing; sign refinement disabled')
            self._sign_labels = set()

        self._detection_interval = max(0.0, detection_interval)
        self._display_ttl = max(0.05, display_ttl)
        self._sign_cache_ttl = max(0.1, sign_cache_ttl)

        self._last_timestamp = -float('inf')
        self._last_snapshot: Optional[DetectionSnapshot] = None
        self._last_nonempty_snapshot: Optional[DetectionSnapshot] = None
        self._sign_cache: List[SignCacheEntry] = []

    @staticmethod
    def _normalise_yolo_names(names: Sequence[str] | dict) -> dict:
        if isinstance(names, dict):
            return {int(idx): value for idx, value in names.items()}
        return {idx: value for idx, value in enumerate(names)}

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Tuple[DetectionItem, ...]]:
        """Run detection for a frame and return annotated copy & detections."""
        if timestamp - self._last_timestamp < self._detection_interval and self._last_snapshot is not None:
            detections = self._resolve_display_items(timestamp)
            annotated = self._render_tracks(frame_bgr.copy(), detections)
            return annotated, detections

        results = self._yolo(frame_bgr, conf=self._conf, iou=self._iou, verbose=False)
        items: Tuple[DetectionItem, ...] = tuple()
        if results:
            items = self._build_items(results[0], frame_bgr, timestamp)

        snapshot = DetectionSnapshot(items=items, timestamp=timestamp)
        self._last_snapshot = snapshot
        if items:
            self._last_nonempty_snapshot = snapshot
        self._last_timestamp = timestamp

        detections = self._resolve_display_items(timestamp)
        annotated = self._render_tracks(frame_bgr.copy(), detections)
        return annotated, detections

    def _resolve_display_items(self, timestamp: float) -> Tuple[DetectionItem, ...]:
        snapshot = self._last_snapshot
        last_nonempty = self._last_nonempty_snapshot
        if snapshot is None and last_nonempty is None:
            return tuple()
        if snapshot is not None:
            age = timestamp - snapshot.timestamp
            if snapshot.items and age <= self._display_ttl:
                return snapshot.items
            if not snapshot.items and last_nonempty is not None:
                nonempty_age = timestamp - last_nonempty.timestamp
                if nonempty_age <= self._display_ttl:
                    return last_nonempty.items
        if last_nonempty is not None:
            nonempty_age = timestamp - last_nonempty.timestamp
            if nonempty_age <= self._display_ttl:
                return last_nonempty.items
        return tuple()

    def _render_tracks(self, frame_bgr: np.ndarray, detections: Sequence[DetectionItem]) -> np.ndarray:
        tracker_inputs = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            tracker_inputs.append((np.array([x1, y1, x2, y2], dtype=np.float32), detection.text, 1.0, detection.color))

        tracks = self._tracker.update(tracker_inputs)

        if not tracks and tracker_inputs:
            return self._draw_raw(frame_bgr, detections)

        if not tracks:
            return frame_bgr

        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), track.color, 2)
            label = f"ID {track.track_id}: {track.label}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_w, label_h = label_size
            label_y = max(0, y1 - label_h - 6)
            cv2.rectangle(frame_bgr, (x1, label_y), (x1 + label_w + 10, label_y + label_h + 6), (20, 20, 20), -1)
            cv2.putText(frame_bgr, label, (x1 + 5, label_y + label_h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame_bgr

    def _build_items(self, result, frame_bgr: np.ndarray, timestamp: float) -> Tuple[DetectionItem, ...]:
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return tuple()
        frame_height, frame_width = frame_bgr.shape[:2]
        np_boxes = boxes.xyxy.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        items: List[DetectionItem] = []

        for bbox, score, cls_idx in zip(np_boxes, scores, classes):
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            x2 = max(0, min(frame_width - 1, x2))
            y2 = max(0, min(frame_height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            raw_label = self._names.get(int(cls_idx), f'class_{cls_idx}')
            label_lower = raw_label.lower()
            text = f"{raw_label} {score:.2f}"
            color = BOX_COLOR_MAP.get(label_lower, DEFAULT_BOX_COLOR)

            if label_lower in self._sign_labels and self._resnet_bundle is not None:
                classified = self._classify_sign(frame_bgr, (x1, y1, x2, y2), timestamp)
                if classified is not None:
                    sign_label, confidence = classified
                    text = f"{sign_label} {confidence:.2f}"
                    color = BOX_COLOR_MAP.get('traffic sign', DEFAULT_BOX_COLOR)

            items.append(DetectionItem(bbox=(x1, y1, x2, y2), text=text, color=color))
        return tuple(items)

    def _classify_sign(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        timestamp: float,
    ) -> Optional[Tuple[str, float]]:
        cached = self._lookup_sign_cache(bbox, timestamp)
        if cached is not None:
            return cached.label, cached.confidence
        if self._resnet_bundle is None:
            return None

        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tensor = self._resnet_bundle.transform(pil_image).unsqueeze(0).to(self._resnet_bundle.device)
        with torch.no_grad():
            logits = self._resnet_bundle.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        label = RESNET_CLASS_NAMES[predicted_idx.item()]
        confidence_value = float(confidence.item())
        self._update_sign_cache(bbox, label, confidence_value, timestamp)
        return label, confidence_value

    def _lookup_sign_cache(self, bbox: Tuple[int, int, int, int], timestamp: float) -> Optional[SignCacheEntry]:
        if not self._sign_cache:
            return None
        retained: List[SignCacheEntry] = []
        best_entry: Optional[SignCacheEntry] = None
        best_iou = 0.0
        for entry in self._sign_cache:
            if timestamp - entry.timestamp > self._sign_cache_ttl:
                continue
            retained.append(entry)
            iou = self._compute_iou(bbox, entry.bbox)
            if iou > 0.55 and iou > best_iou:
                best_entry = entry
                best_iou = iou
        self._sign_cache = retained[-32:]
        return best_entry

    def _update_sign_cache(self, bbox: Tuple[int, int, int, int], label: str, confidence: float, timestamp: float) -> None:
        entry = SignCacheEntry(bbox=bbox, label=label, confidence=confidence, timestamp=timestamp)
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
    def _draw_raw(frame_bgr: np.ndarray, detections: Sequence[DetectionItem]) -> np.ndarray:
        for item in detections:
            x1, y1, x2, y2 = item.bbox
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), item.color, 2)
            label_size, _ = cv2.getTextSize(item.text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_w, label_h = label_size
            label_y = max(0, y1 - label_h - 6)
            cv2.rectangle(frame_bgr, (x1, label_y), (x1 + label_w + 10, label_y + label_h + 6), (20, 20, 20), -1)
            cv2.putText(frame_bgr, item.text, (x1 + 5, label_y + label_h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame_bgr


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run YOLO + ResNet post-analysis over a recorded CARLA video')
    parser.add_argument('video', help='path to the input video file (.mp4, .avi, etc.)')
    parser.add_argument('--weights', default='best.pt', help='path to YOLOv11 weights (default: best.pt)')
    parser.add_argument('--resnet', default='best_traffic_sign_classifier_advanced.pth', help='path to ResNet sign classifier weights (default: best_traffic_sign_classifier_advanced.pth; use "none" to disable)')
    parser.add_argument('--output', default=None, help='optional path to save the annotated video')
    parser.add_argument('--conf', default=0.25, type=float, help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--iou', default=0.45, type=float, help='YOLO IoU threshold (default: 0.45)')
    parser.add_argument('--device', default=None, help='Torch device for inference (e.g., cuda:0)')
    parser.add_argument('--sign-labels', default=None, help='comma-separated YOLO class names to refine with ResNet (default uses known traffic-sign labels)')
    parser.add_argument('--detection-interval', default=0.02, type=float, help='minimum seconds between consecutive YOLO runs (default: 0.02)')
    parser.add_argument('--display-ttl', default=0.3, type=float, help='seconds to keep detections visible without refresh (default: 0.3)')
    parser.add_argument('--sign-cache-ttl', default=0.75, type=float, help='seconds to reuse cached ResNet predictions (default: 0.75)')
    parser.add_argument('--no-display', action='store_true', help='disable on-screen display (useful for batch processing)')
    parser.add_argument('--debug', action='store_true', help='enable verbose logging')
    return parser.parse_args(argv)


def create_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if output_path.lower().endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer for output file: {output_path}')
    return writer


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    video_path = args.video
    if not os.path.exists(video_path):
        logging.error('Input video not found: %s', video_path)
        return 1

    sign_labels = normalise_sign_labels(
        None if args.sign_labels is None else args.sign_labels.split(',')
    )
    resnet_path = None if args.resnet.lower() == 'none' else args.resnet

    pipeline = VideoDetectionPipeline(
        yolo_weights=args.weights,
        resnet_weights=resnet_path,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        sign_labels=sign_labels,
        detection_interval=args.detection_interval,
        display_ttl=args.display_ttl,
        sign_cache_ttl=args.sign_cache_ttl,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error('Could not open video: %s', video_path)
        return 1

    writer = None
    if args.output:
        writer = create_writer(cap, args.output)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    paused = False

    logging.info('Processing video %s (fps=%.2f)', video_path, fps)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logging.info('End of video reached')
                    break
                timestamp = frame_idx / fps
                annotated, detections = pipeline.process_frame(frame, timestamp)

                if writer is not None:
                    writer.write(annotated)

                if not args.no_display:
                    overlay = _compose_overlay(annotated, detections, timestamp)
                    cv2.imshow('CARLA Post Analysis', overlay)

                frame_idx += 1

            key = cv2.waitKey(1 if not args.no_display else 0) & 0xFF
            if key == ord('q'):
                logging.info('User requested exit')
                break
            if key == ord(' '):
                paused = not paused
            if key == ord('s'):
                paused = True
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


def _compose_overlay(frame: np.ndarray, detections: Sequence[DetectionItem], timestamp: float) -> np.ndarray:
    overlay = frame.copy()
    text = f"Frame time: {timestamp:06.2f}s | Detections: {len(detections)}"
    cv2.rectangle(overlay, (10, 10), (10 + 320, 40), (0, 0, 0), -1)
    cv2.putText(overlay, text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return overlay


if __name__ == '__main__':
    sys.exit(run())
