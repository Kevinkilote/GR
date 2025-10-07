"""Simple IOU-based tracker for associating detections across frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    label: str
    confidence: float
    color: Tuple[int, int, int]
    ttl: int
    streak: int
    age: int
    last_seen: int


class IOUTracker:
    """Assigns stable IDs to detections using IoU and greedy matching."""

    def __init__(
        self,
        max_ttl: int = 10,
        min_streak: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_ttl = max_ttl
        self.min_streak = min_streak
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._frame_count = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0

    def update(self, detections: Iterable[Tuple[np.ndarray, str, float, Tuple[int, int, int]]]) -> List[Track]:
        self._frame_count += 1
        det_list = list(detections)
        updated_tracks: Dict[int, Track] = {}

        if self._tracks and det_list:
            track_ids = list(self._tracks.keys())
            track_boxes = np.array([self._tracks[t].bbox for t in track_ids])
            det_boxes = np.array([d[0] for d in det_list])
            iou_matrix = compute_iou(track_boxes, det_boxes)

            assigned_tracks = set()
            assigned_dets = set()

            while True:
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[idx]
                if best_iou < self.iou_threshold:
                    break
                track_idx, det_idx = idx
                track_id = track_ids[track_idx]
                if track_id in assigned_tracks or det_idx in assigned_dets:
                    iou_matrix[track_idx, det_idx] = -1
                    continue
                assigned_tracks.add(track_id)
                assigned_dets.add(det_idx)
                iou_matrix[track_idx, :] = -1
                iou_matrix[:, det_idx] = -1
                det_box, det_label, det_conf, det_color = det_list[det_idx]
                prev_track = self._tracks[track_id]
                updated_tracks[track_id] = Track(
                    track_id=track_id,
                    bbox=det_box,
                    label=det_label,
                    confidence=det_conf,
                    color=det_color,
                    ttl=self.max_ttl,
                    streak=prev_track.streak + 1,
                    age=prev_track.age + 1,
                    last_seen=self._frame_count,
                )
            remaining_dets = [i for i in range(len(det_list)) if i not in assigned_dets]
        else:
            remaining_dets = list(range(len(det_list)))

        for det_idx in remaining_dets:
            det_box, det_label, det_conf, det_color = det_list[det_idx]
            track_id = self._next_id
            self._next_id += 1
            updated_tracks[track_id] = Track(
                track_id=track_id,
                bbox=det_box,
                label=det_label,
                confidence=det_conf,
                color=det_color,
                ttl=self.max_ttl,
                streak=1,
                age=1,
                last_seen=self._frame_count,
            )

        for track_id, track in self._tracks.items():
            if track_id in updated_tracks:
                continue
            ttl = track.ttl - 1
            if ttl <= 0:
                continue
            updated_tracks[track_id] = Track(
                track_id=track_id,
                bbox=track.bbox,
                label=track.label,
                confidence=track.confidence,
                color=track.color,
                ttl=ttl,
                streak=track.streak,
                age=track.age + 1,
                last_seen=track.last_seen,
            )

        self._tracks = updated_tracks
        return [track for track in self._tracks.values() if track.streak >= self.min_streak]


def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    ax1 = boxes_a[:, 0][:, None]
    ay1 = boxes_a[:, 1][:, None]
    ax2 = boxes_a[:, 2][:, None]
    ay2 = boxes_a[:, 3][:, None]
    bx1 = boxes_b[:, 0][None, :]
    by1 = boxes_b[:, 1][None, :]
    bx2 = boxes_b[:, 2][None, :]
    by2 = boxes_b[:, 3][None, :]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    union = np.where(union <= 0, 1e-6, union)

    return inter_area / union
