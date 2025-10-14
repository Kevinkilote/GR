#!/usr/bin/env python3
"""Lightweight post-run analysis viewer with YOLOv11 + ResNet classification."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from speed_limit_ocr import SpeedLimitOCR
from traffic_sign_recognition import (
    DEFAULT_SIGN_LABELS,
    OTHER_SIGN_LABEL,
    RESNET_CLASS_NAMES,
    ResNetBundle,
    load_resnet,
    normalise_sign_labels,
)


@dataclass(frozen=True)
class DetectionLabel:
    location: Tuple[int, int]
    text: str


@dataclass(frozen=True)
class DetectionSnapshot:
    items: Tuple[DetectionLabel, ...]
    timestamp: float


@dataclass(frozen=True)
class SignCacheEntry:
    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float
    timestamp: float


class PlaybackClock:
    """Plays frames in sync with the source timestamps (optionally scaled)."""

    def __init__(self, fps: float, speed: float = 1.0) -> None:
        self._fps = fps if fps > 0 else 0.0
        self._speed = max(1e-6, speed)
        self._default_interval = (1.0 / self._fps) / self._speed if self._fps > 0 else None
        self._anchor_video_time: Optional[float] = None
        self._anchor_wall_time: Optional[float] = None
        self._last_interval: Optional[float] = None

    def align(self, video_time: float) -> None:
        if self._anchor_video_time is None or self._anchor_wall_time is None:
            self._anchor_video_time = video_time
            self._anchor_wall_time = time.perf_counter()
            return
        target_wall = self._anchor_wall_time + (video_time - self._anchor_video_time) / self._speed
        now = time.perf_counter()
        remaining = target_wall - now
        if remaining > 0:
            time.sleep(remaining)
        interval = max(0.0, video_time - self._anchor_video_time)
        if interval > 0:
            self._last_interval = interval / self._speed
            self._anchor_video_time = video_time
            self._anchor_wall_time = target_wall if remaining > 0 else time.perf_counter()

    def key_delay_ms(self, paused: bool) -> int:
        if paused:
            return 0
        interval = self._last_interval
        if interval is None:
            interval = self._default_interval
        if interval is None or interval <= 0:
            return 1
        return max(1, int(round(interval * 1000)))

    def reset(self) -> None:
        self._anchor_video_time = None
        self._anchor_wall_time = None
        self._last_interval = None


class LabelDetector:
    """Runs YOLO detections and optional ResNet refinement, caching results."""

    def __init__(
        self,
        *,
        yolo_weights: str,
        resnet_weights: Optional[str],
        conf: float,
        iou: float,
        device: Optional[str],
        detection_interval: float,
        display_ttl: float,
        sign_labels: Iterable[str],
        sign_cache_ttl: float,
        sign_conf_threshold: float,
        skip_labels: Iterable[str],
    ) -> None:
        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
        self._yolo = YOLO(yolo_weights)
        self._yolo_device = device
        self._conf = conf
        self._iou = iou
        self._detection_interval = max(0.0, detection_interval)
        self._display_ttl = max(0.05, display_ttl)
        self._sign_cache_ttl = max(0.1, sign_cache_ttl)
        self._sign_conf_threshold = max(0.0, sign_conf_threshold)
        self._names = self._normalise_names(self._yolo.names)
        self._skip_labels = {label.lower() for label in skip_labels}

        self._sign_labels = {label.lower() for label in sign_labels}
        self._resnet_bundle: Optional[ResNetBundle] = None
        if self._sign_labels and resnet_weights:
            self._resnet_bundle = load_resnet(resnet_weights, device_hint=device)
        elif self._sign_labels:
            logging.warning(
                "Sign labels requested but ResNet weights missing; disabling refinement"
            )
            self._sign_labels = set()

        self._speed_limit_ocr = SpeedLimitOCR()
        self._sign_cache: List[SignCacheEntry] = []
        self._last_inference_time = -float("inf")
        self._last_snapshot: Optional[DetectionSnapshot] = None
        self._last_nonempty_snapshot: Optional[DetectionSnapshot] = None

    @staticmethod
    def _normalise_names(names: Sequence[str] | dict) -> dict:
        if isinstance(names, dict):
            return {int(idx): value for idx, value in names.items()}
        return {idx: value for idx, value in enumerate(names)}

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Tuple[DetectionLabel, ...]:
        needs_inference = timestamp - self._last_inference_time >= self._detection_interval
        if needs_inference:
            detections = self._run_inference(frame_bgr, timestamp)
            snapshot = DetectionSnapshot(items=detections, timestamp=timestamp)
            self._last_snapshot = snapshot
            if detections:
                self._last_nonempty_snapshot = snapshot
            self._last_inference_time = timestamp
        return self._resolve_items(timestamp)

    def _run_inference(self, frame_bgr: np.ndarray, timestamp: float) -> Tuple[DetectionLabel, ...]:
        inference_kwargs = dict(conf=self._conf, iou=self._iou, verbose=False)
        if self._yolo_device:
            inference_kwargs["device"] = self._yolo_device
        results = self._yolo(frame_bgr, **inference_kwargs)
        if not results:
            return tuple()
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return tuple()

        np_boxes = boxes.xyxy.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        frame_height, frame_width = frame_bgr.shape[:2]

        labels: List[DetectionLabel] = []
        for bbox, score, cls_idx in zip(np_boxes, scores, classes):
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            x2 = max(0, min(frame_width - 1, x2))
            y2 = max(0, min(frame_height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            raw_label = self._names.get(int(cls_idx), f"class_{cls_idx}")
            label_lower = raw_label.lower()
            if label_lower in self._skip_labels:
                continue
            text = f"{raw_label} {score:.2f}"

            if (
                label_lower in self._sign_labels
                and self._resnet_bundle is not None
            ):
                refined = self._classify_sign(frame_bgr, (x1, y1, x2, y2), timestamp)
                if refined is not None:
                    sign_label, confidence = refined
                    display = self._format_sign_label(sign_label)
                    if sign_label == OTHER_SIGN_LABEL:
                        text = display
                    elif sign_label.startswith("speed-limit"):
                        text = f"{display} ({confidence:.2f})"
                    else:
                        text = f"{display} {confidence:.2f}"

            labels.append(DetectionLabel(location=(x1, max(0, y1 - 8)), text=text))
        return tuple(labels)

    def _resolve_items(self, timestamp: float) -> Tuple[DetectionLabel, ...]:
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
            confidence, idx = torch.max(probabilities, dim=1)
        label = RESNET_CLASS_NAMES[idx.item()]
        confidence_value = float(confidence.item())

        if "maximum-speed-limit" in label or label == OTHER_SIGN_LABEL:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ocr = self._speed_limit_ocr.infer(crop_rgb)
            if ocr is not None:
                label = f"speed-limit-{ocr.value}"
                confidence_value = min(confidence_value, ocr.score)
            elif "maximum-speed-limit" in label:
                label = "speed-limit"

        if confidence_value < self._sign_conf_threshold:
            label = OTHER_SIGN_LABEL
        self._update_sign_cache(bbox, label, confidence_value, timestamp)
        return label, confidence_value

    def _lookup_sign_cache(
        self,
        bbox: Tuple[int, int, int, int],
        timestamp: float,
    ) -> Optional[SignCacheEntry]:
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
        if best_entry and best_entry.label == OTHER_SIGN_LABEL:
            return None
        return best_entry

    def _update_sign_cache(
        self,
        bbox: Tuple[int, int, int, int],
        label: str,
        confidence: float,
        timestamp: float,
    ) -> None:
        entry = SignCacheEntry(bbox=bbox, label=label, confidence=confidence, timestamp=timestamp)
        self._sign_cache.append(entry)
        if len(self._sign_cache) > 64:
            self._sign_cache = self._sign_cache[-64:]

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
            return "Other Sign"
        if label == "speed-limit":
            return "Speed Limit"
        if label.startswith("speed-limit-"):
            suffix = label.split("-", 2)[2]
            return f"Speed Limit {suffix}"
        parts = label.replace("--", " ").replace("-", " ").replace("_", " ").split()
        return " ".join(part.capitalize() for part in parts)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a recorded CARLA video with YOLO + ResNet labels.")
    parser.add_argument("video", help="path to the input video file")
    parser.add_argument("--weights", default="best.pt", help="path to YOLOv11 weights (default: best.pt)")
    parser.add_argument("--resnet", default="best_traffic_sign_classifier_advanced.pth", help="path to ResNet weights (use 'none' to disable)")
    parser.add_argument("--conf", default=0.25, type=float, help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--iou", default=0.45, type=float, help="YOLO IoU threshold (default: 0.45)")
    parser.add_argument("--device", default=None, help="Torch device string for inference (e.g., 'cuda:0')")
    parser.add_argument("--sign-labels", default=None, help="comma-separated YOLO class names to refine with ResNet")
    parser.add_argument("--detection-interval", default=0.05, type=float, help="minimum seconds between YOLO runs (default: 0.05)")
    parser.add_argument("--display-ttl", default=0.3, type=float, help="seconds to keep last detections visible (default: 0.3)")
    parser.add_argument("--sign-cache-ttl", default=0.75, type=float, help="seconds to reuse cached ResNet predictions (default: 0.75)")
    parser.add_argument("--sign-conf-threshold", default=0.6, type=float, help="minimum ResNet confidence before accepting a label (default: 0.6)")
    parser.add_argument("--skip-labels", default="", help="comma-separated YOLO class names to ignore")
    parser.add_argument("--playback-speed", default=1.0, type=float, help="playback speed multiplier (default: 1.0)")
    parser.add_argument("--no-display", action="store_true", help="disable on-screen display (process only)")
    parser.add_argument("--debug", action="store_true", help="enable verbose logging")
    return parser.parse_args(argv)


def overlay_labels(frame_bgr: np.ndarray, detections: Sequence[DetectionLabel]) -> np.ndarray:
    annotated = frame_bgr.copy()
    for item in detections:
        x, y = item.location
        text = item.text
        if not text:
            continue
        cv2.putText(annotated, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    video_path = args.video
    if not os.path.exists(video_path):
        logging.error("Input video not found: %s", video_path)
        return 1

    resnet_path = None if args.resnet.lower() == "none" else args.resnet
    sign_labels = normalise_sign_labels(
        None if args.sign_labels is None else args.sign_labels.split(",")
    )
    skip_labels = {label.strip().lower() for label in args.skip_labels.split(",") if label.strip()} if args.skip_labels else set()

    detector = LabelDetector(
        yolo_weights=args.weights,
        resnet_weights=resnet_path,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        detection_interval=args.detection_interval,
        display_ttl=args.display_ttl,
        sign_labels=sign_labels if sign_labels else DEFAULT_SIGN_LABELS,
        sign_cache_ttl=args.sign_cache_ttl,
        sign_conf_threshold=args.sign_conf_threshold,
        skip_labels=skip_labels,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video: %s", video_path)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clock = PlaybackClock(fps, args.playback_speed)

    frame_idx = 0
    paused = False
    last_display: Optional[np.ndarray] = None

    logging.info("Processing %s (fps=%.2f)", video_path, fps)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logging.info("End of video reached")
                    break
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if timestamp_ms and timestamp_ms >= 0:
                    timestamp = timestamp_ms / 1000.0
                else:
                    timestamp = frame_idx / fps if fps > 0 else 0.0
                clock.align(timestamp)
                detections = detector.process(frame, timestamp)
                display = overlay_labels(frame, detections)
                frame_idx += 1
                last_display = display
                if not args.no_display:
                    cv2.imshow("Analysis", display)
            key_delay = clock.key_delay_ms(paused)
            key = cv2.waitKey(key_delay) & 0xFF if not args.no_display else -1
            if key == ord("q"):
                logging.info("User requested exit")
                break
            if key == ord(" "):
                paused = not paused
                clock.reset()
                continue
            if key == ord("s"):
                paused = True
                clock.reset()
            if paused:
                if args.no_display:
                    time.sleep(0.05)
                elif last_display is not None:
                    cv2.imshow("Analysis", last_display)
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
