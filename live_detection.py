#!/usr/bin/env python3
"""CARLA manual control with optional YOLOv11 live detection overlay."""
from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pygame

import manual_control_steeringwheel as base
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

from tracker import IOUTracker
from speed_limit_ocr import SpeedLimitOCR
from traffic_sign_recognition import DEFAULT_SIGN_LABELS, OTHER_SIGN_LABEL, RESNET_CLASS_NAMES


BOX_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    'traffic sign': (0, 255, 0),
    'traffic light': (255, 255, 0),
    'vehicle': (0, 170, 255),
    'car': (0, 170, 255),
    OTHER_SIGN_LABEL: (160, 160, 160),
}
DEFAULT_BOX_COLOR: Tuple[int, int, int] = (255, 255, 255)


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
        self._resnet_model = None
        self._resnet_transform = None
        self._torch_device: Optional[torch.device] = None
        self._sign_cache: List[SignCacheEntry] = []
        self._speed_limit_ocr = SpeedLimitOCR()

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
            text = f"{raw_label} {score:.2f}"
            color = self._resolve_color(label_lower)
            if label_lower in self._sign_labels and self._resnet_model is not None and self._resnet_transform is not None:
                classified = self._classify_sign(frame_rgb, (x1, y1, x2, y2), timestamp)
                if classified is not None:
                    sign_label, confidence = classified
                    display_label = self._format_sign_label(sign_label)
                    if sign_label == OTHER_SIGN_LABEL:
                        text = display_label
                        color = self._resolve_color(OTHER_SIGN_LABEL)
                    elif sign_label.startswith('speed-limit-'):
                        text = f"{display_label} ({confidence:.2f})"
                        color = self._resolve_color('speed-limit')
                    else:
                        text = f"{display_label} {confidence:.2f}"
                        color = self._resolve_color('traffic sign')
            items.append(DetectionItem(bbox=(x1, y1, x2, y2), text=text, color=color))
        return tuple(items)

    @staticmethod
    def _resolve_color(label_lower: str) -> Tuple[int, int, int]:
        if label_lower.startswith('speed-limit'):
            return BOX_COLOR_MAP.get('traffic sign', DEFAULT_BOX_COLOR)
        return BOX_COLOR_MAP.get(label_lower, DEFAULT_BOX_COLOR)

    def _classify_sign(self, frame_rgb: np.ndarray, bbox: Tuple[int, int, int, int], timestamp: float) -> Optional[Tuple[str, float]]:
        if self._resnet_model is None or self._resnet_transform is None or self._torch_device is None:
            return None
        cached = self._lookup_sign_cache(bbox, timestamp)
        if cached is not None:
            return cached.label, cached.confidence
        x1, y1, x2, y2 = bbox
        cropped = frame_rgb[y1:y2, x1:x2]
        if cropped.size == 0:
            return None
        image = Image.fromarray(cropped)
        input_tensor = self._resnet_transform(image).unsqueeze(0).to(self._torch_device)
        with torch.no_grad():
            output = self._resnet_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        label = RESNET_CLASS_NAMES[predicted_idx.item()]
        confidence_value = float(confidence.item())

        use_ocr = 'maximum-speed-limit' in label or label == OTHER_SIGN_LABEL
        ocr_result = None
        if use_ocr and self._speed_limit_ocr is not None:
            crop_rgb = frame_rgb[y1:y2, x1:x2]
            ocr_result = self._speed_limit_ocr.infer(crop_rgb)
        if ocr_result:
            digits, ocr_score = ocr_result
            label = f'speed-limit-{digits}'
            confidence_value = ocr_score

        if label.startswith('speed-limit-'):
            if confidence_value < self.sign_conf_threshold:
                return OTHER_SIGN_LABEL, confidence_value
            self._update_sign_cache(bbox, label, confidence_value, timestamp)
            return label, confidence_value

        if confidence_value < self.sign_conf_threshold:
            return OTHER_SIGN_LABEL, confidence_value
        self._update_sign_cache(bbox, label, confidence_value, timestamp)
        return label, confidence_value

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
    def _format_sign_label(label: str) -> str:
        if label == OTHER_SIGN_LABEL:
            return 'Other Sign'
        if label.startswith('speed-limit-'):
            return f"Speed Limit {label.split('-', 2)[2].upper()}" if label.count('-') >= 2 else 'Speed Limit'
        parts = label.replace('--', ' ').replace('-', ' ').replace('_', ' ').split()
        return ' '.join(part.capitalize() for part in parts)



class LiveDetectionCameraManager(base.CameraManager):
    """Camera manager that overlays YOLO detections when enabled."""

    BOX_WIDTH = 2
    BG_COLOR = (20, 20, 20)
    LABEL_CACHE_MAX = 64

    def __init__(self, parent_actor, hud, detection: DetectionContext):
        super().__init__(parent_actor, hud)
        self._detection = detection
        self._label_font = pygame.font.Font(pygame.font.get_default_font(), 18)
        self._surface_size: Optional[Tuple[int, int]] = None
        self._label_cache: Dict[str, pygame.Surface] = {}
        self._tracker = IOUTracker(max_ttl=15, min_streak=1, iou_threshold=0.3)

    def set_sensor(self, index, notify=True):  # noqa: D401 (interface inherited)
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent,
            )
            weak_self = base.weakref.ref(self)
            self.sensor.listen(lambda image: LiveDetectionCameraManager._parse_image(weak_self, image))
            self._tracker.reset()
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
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
        else:
            self._tracker.update([])
        self.surface = surface

    def _ensure_surface(self, width: int, height: int) -> pygame.Surface:
        size = (width, height)
        if self.surface is None or self._surface_size != size:
            self.surface = pygame.Surface(size).convert()
            self._surface_size = size
        return self.surface

    def _draw_detections(self, surface: pygame.Surface, detections: Sequence[DetectionItem]) -> None:
        width, height = surface.get_width(), surface.get_height()
        tracker_inputs = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            tracker_inputs.append((np.array([x1, y1, x2, y2], dtype=np.float32), detection.text, 1.0, detection.color))

        tracks = self._tracker.update(tracker_inputs)

        if not tracks and tracker_inputs:
            self._draw_raw_detections(surface, tracker_inputs)
            return

        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
            pygame.draw.rect(surface, track.color, rect, self.BOX_WIDTH)
            label = f"ID {track.track_id}: {track.label}"
            self._draw_label(surface, rect, label)

    def _draw_raw_detections(self, surface: pygame.Surface, tracker_inputs) -> None:
        for det_box, det_label, _conf, det_color in tracker_inputs:
            x1, y1, x2, y2 = det_box.astype(int)
            rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
            pygame.draw.rect(surface, det_color, rect, self.BOX_WIDTH)
            self._draw_label(surface, rect, det_label)

    def _draw_label(self, surface: pygame.Surface, rect: pygame.Rect, label: str) -> None:
        label_surface = self._get_label_surface(label)
        surface.blit(label_surface, (rect.x, max(0, rect.y - label_surface.get_height())))

    def _get_label_surface(self, label: str) -> pygame.Surface:
        cached = self._label_cache.get(label)
        if cached is None:
            text_surface = self._label_font.render(label, True, (255, 255, 255))
            label_surface = pygame.Surface((text_surface.get_width() + 6, text_surface.get_height() + 6), pygame.SRCALPHA)
            label_surface.fill((*self.BG_COLOR, 200))
            label_surface.blit(text_surface, (3, 3))
            if len(self._label_cache) >= self.LABEL_CACHE_MAX:
                self._label_cache.clear()
            self._label_cache[label] = label_surface
            cached = label_surface
        return cached.copy()


class LiveDetectionWorld(base.World):
    """World wrapper that swaps in a detection-aware camera manager."""

    def __init__(self, carla_world, hud, actor_filter, detection: DetectionContext):
        self._detection = detection
        super().__init__(carla_world, hud, actor_filter)

    def restart(self):
        super().restart()
        self._attach_detection_camera()

    def _attach_detection_camera(self) -> None:
        previous_manager = getattr(self, 'camera_manager', None)
        transform_index = 0
        sensor_index = 0
        if previous_manager is not None:
            transform_index = previous_manager.transform_index
            sensor_index = previous_manager.index or 0
            if previous_manager.sensor is not None:
                previous_manager.sensor.stop()
                previous_manager.sensor.destroy()
        self.camera_manager = LiveDetectionCameraManager(self.player, self.hud, self._detection)
        self.camera_manager.transform_index = transform_index
        self.camera_manager.set_sensor(sensor_index, notify=False)


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
    )
    try:
        client = base.carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        hud = base.HUD(args.width, args.height)
        world = LiveDetectionWorld(client.get_world(), hud, args.filter, detection_context)
        controller = LiveDualControl(world, args.autopilot, detection_context, args.detect_button)
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
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
    parser.add_argument('--conf', default=0.25, type=float, help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--iou', default=0.45, type=float, help='YOLO IoU threshold (default: 0.45)')
    parser.add_argument('--device', default=None, help='Torch device for YOLO (e.g., cuda:0)')
    parser.add_argument('--detection-interval', default=0.05, type=float, help='minimum seconds between YOLO inferences (default: 0.05)')
    parser.add_argument('--display-ttl', default=0.3, type=float, help='seconds to keep last detections on screen (default: 0.3)')
    parser.add_argument('--sign-cache-ttl', default=0.75, type=float, help='seconds to reuse cached ResNet classifications (default: 0.75)')
    parser.add_argument('--sign-conf-threshold', default=0.6, type=float, help='minimum ResNet confidence before accepting a sign class (default: 0.6)')
    parser.add_argument('--detect-button', default=None, type=int, help='joystick button index to toggle detection (omit to use keyboard only; set -1 to disable)')
    parser.add_argument('--debug', action='store_true', help='print debug information')
    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    if args.detect_button is not None and args.detect_button < 0:
        args.detect_button = None
    if args.resnet and args.resnet.strip().lower() == 'none':
        args.resnet = None
    args.sign_labels = parse_sign_labels_arg(args.sign_labels)
    args.display_ttl = max(0.05, args.display_ttl)
    args.sign_cache_ttl = max(0.1, args.sign_cache_ttl)
    args.sign_conf_threshold = max(0.0, args.sign_conf_threshold)
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
