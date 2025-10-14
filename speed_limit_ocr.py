"""Shape-aware OCR for circular speed limit signs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class OCRResult:
    value: str
    score: float


class SpeedLimitOCR:
    """Recognise the numeric value displayed on a circular speed-limit sign."""

    def __init__(
        self,
        *,
        target_size: int = 192,
        min_ring_ratio: float = 0.08,
        max_ring_ratio: float = 0.45,
        min_digit_area: float = 0.01,
        digit_template_size: Tuple[int, int] = (48, 32),
        digit_threshold: float = 0.45,
        candidate_digits: Iterable[str] = tuple(str(d) for d in range(10)),
    ) -> None:
        self.target_size = target_size
        self.min_ring_ratio = min_ring_ratio
        self.max_ring_ratio = max_ring_ratio
        self.min_digit_area = min_digit_area
        self.digit_template_size = digit_template_size
        self.digit_threshold = digit_threshold
        self.digit_templates: Dict[str, np.ndarray] = self._build_digit_templates(candidate_digits)

    def infer(self, crop_rgb: np.ndarray) -> Optional[OCRResult]:
        if crop_rgb is None or crop_rgb.size == 0:
            return None

        prepared = self._prepare_image(crop_rgb)
        hsv = cv2.cvtColor(prepared, cv2.COLOR_RGB2HSV)
        ring_mask, inner_mask = self._extract_ring_masks(hsv)
        if ring_mask is None or inner_mask is None:
            return None

        ring_area = float(cv2.countNonZero(ring_mask))
        interior_area = float(cv2.countNonZero(inner_mask))
        if interior_area <= 0:
            return None
        ratio = ring_area / interior_area
        if ratio < self.min_ring_ratio or ratio > self.max_ring_ratio:
            return None

        digit_mask = self._extract_digit_mask(prepared, inner_mask)
        if digit_mask is None:
            return None
        digits = self._recognise_digits(digit_mask)
        if not digits:
            return None

        value_str = "".join(d for d, _score in digits if d)
        if not value_str:
            return None
        try:
            numeric = str(int(value_str))
        except ValueError:
            return None
        confidence = float(np.mean([score for _digit, score in digits]))
        return OCRResult(value=numeric, score=confidence)

    def _prepare_image(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Pad to square and resize to the target size."""
        h, w = crop_rgb.shape[:2]
        if h == 0 or w == 0:
            return crop_rgb
        side = max(h, w)
        pad_top = (side - h) // 2
        pad_bottom = side - h - pad_top
        pad_left = (side - w) // 2
        pad_right = side - w - pad_left
        padded = cv2.copyMakeBorder(
            crop_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
        prepared = cv2.resize(padded, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        return prepared

    def _extract_ring_masks(self, hsv: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return binary masks for the red ring and interior."""
        lower_red1 = np.array([0, 90, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 90, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        if cv2.countNonZero(red_mask) < 50:
            return None, None

        red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
        red_mask = cv2.threshold(red_mask, 0, 255, cv2.THRESH_BINARY)[1]
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 300:
            return None, None

        ring_mask = np.zeros_like(red_mask)
        cv2.drawContours(ring_mask, [largest], -1, 255, thickness=cv2.FILLED)

        radius = int(math.sqrt(area / math.pi))
        moments = cv2.moments(largest)
        if abs(moments["m00"]) < 1e-5:
            center = (self.target_size // 2, self.target_size // 2)
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            center = (cx, cy)

        inner_radius = max(1, int(radius * 0.65))
        inner_mask = np.zeros_like(ring_mask)
        cv2.circle(inner_mask, center, inner_radius, 255, thickness=cv2.FILLED)
        ring_only = cv2.subtract(ring_mask, cv2.erode(ring_mask, np.ones((9, 9), np.uint8), iterations=1))
        if cv2.countNonZero(ring_only) < 50:
            ring_only = cv2.subtract(ring_mask, inner_mask)
        return ring_only, inner_mask

    def _extract_digit_mask(self, image_rgb: np.ndarray, interior_mask: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=interior_mask)
        norm = cv2.normalize(masked, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(norm, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            2,
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        thresh = cv2.bitwise_and(thresh, thresh, mask=interior_mask)
        if cv2.countNonZero(thresh) < 30:
            return None
        return thresh

    def _recognise_digits(self, digit_mask: np.ndarray) -> List[Tuple[str, float]]:
        contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        digit_candidates: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        area_total = digit_mask.shape[0] * digit_mask.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area / area_total < self.min_digit_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w == 0:
                continue
            if h / float(digit_mask.shape[0]) < 0.2:
                continue
            roi = digit_mask[y:y + h, x:x + w]
            digit, score = self._match_digit(roi)
            if digit and score >= self.digit_threshold:
                digit_candidates.append((digit, score, (x, y, w, h)))

        if not digit_candidates:
            return []
        digit_candidates.sort(key=lambda item: item[2][0])
        return [(digit, score) for digit, score, _bbox in digit_candidates]

    def _match_digit(self, roi: np.ndarray) -> Tuple[str, float]:
        resized = cv2.resize(roi, self.digit_template_size, interpolation=cv2.INTER_AREA)
        resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY)[1]
        best_digit = ""
        best_score = -1.0
        for digit, template in self.digit_templates.items():
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_score = score
                best_digit = digit
        return best_digit, best_score

    def _build_digit_templates(self, digits: Iterable[str]) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        width, height = self.digit_template_size
        font = cv2.FONT_HERSHEY_SIMPLEX
        for digit in digits:
            canvas = np.zeros((height, width), dtype=np.uint8)
            text_size = cv2.getTextSize(digit, font, 1.4, 2)[0]
            text_x = max(0, (width - text_size[0]) // 2)
            text_y = max(text_size[1] + (height - text_size[1]) // 2 - 4, text_size[1])
            cv2.putText(canvas, digit, (text_x, text_y), font, 1.4, 255, 2, cv2.LINE_AA)
            templates[digit] = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)[1]
        return templates
