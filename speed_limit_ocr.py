"""Robust OCR for circular speed-limit signs using shape + digit templates."""
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
    """Recognise the numeric value displayed on a circular speed limit sign."""

    def __init__(
        self,
        *,
        valid_values: Sequence[int] = (
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            55, 60, 65, 70, 75, 80, 85, 90, 95,
            100, 110, 120, 130,
        ),
        min_confidence: float = 0.45,
    ) -> None:
        self.valid_values = {str(v) for v in valid_values}
        self.min_confidence = min_confidence
        self.digit_templates = self._build_digit_templates()

    def infer(self, crop_rgb: np.ndarray) -> Optional[OCRResult]:
        if crop_rgb is None or crop_rgb.size == 0:
            return None
        prepared, circle = self._extract_circle(crop_rgb)
        if prepared is None:
            return None
        digits_mask = self._extract_digit_mask(prepared)
        if digits_mask is None:
            return None
        contours = self._find_digit_contours(digits_mask)
        if not contours:
            return None
        recognised = self._recognise_digits(digits_mask, contours)
        if not recognised:
            return None
        value = ''.join(d for d, _ in recognised)
        if value not in self.valid_values:
            return None
        confidence = float(np.mean([score for _, score in recognised]))
        if confidence < self.min_confidence:
            return None
        return OCRResult(value=value, score=confidence)

    @staticmethod
    def _build_digit_templates() -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = (48, 64)
        for digit in '0123456789':
            canvas = np.zeros(size, dtype=np.uint8)
            text_scale = 1.8 if digit != '1' else 1.4
            text_thickness = 4
            text_size = cv2.getTextSize(digit, font, text_scale, text_thickness)[0]
            x = (size[1] - text_size[0]) // 2
            y = (size[0] + text_size[1]) // 2
            cv2.putText(canvas, digit, (x, y), font, text_scale, 255, text_thickness, cv2.LINE_AA)
            templates[digit] = canvas
        return templates

    def _extract_circle(self, crop_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int]]]:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = blur.shape
        min_dim = min(h, w)
        if min_dim < 60:
            blur = cv2.resize(blur, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            h, w = blur.shape
            crop_rgb = cv2.resize(crop_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(h, w) // 4,
            param1=120,
            param2=40,
            minRadius=int(min(h, w) * 0.25),
            maxRadius=int(min(h, w) * 0.6),
        )
        if circles is None:
            return None, None
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        # Ensure circle inside bounds
        if not (r > 0 and x - r >= 0 and y - r >= 0 and x + r < w and y + r < h):
            return None, None

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), int(r * 0.92), 255, thickness=-1)
        interior_radius = int(r * 0.70)
        cv2.circle(mask, (x, y), interior_radius, 255, thickness=-1)
        prepared = cv2.bitwise_and(crop_rgb, crop_rgb, mask=mask)
        return prepared, (x, y, interior_radius)

    def _extract_digit_mask(self, prepared_rgb: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            3,
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        if cv2.countNonZero(thresh) < 30:
            return None
        return thresh

    @staticmethod
    def _find_digit_contours(binary: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        h, w = binary.shape
        area_total = h * w
        candidates: List[np.ndarray] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            area_ratio = area / area_total
            if area_ratio < 0.003 or area_ratio > 0.15:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 2 or bh <= 6:
                continue
            aspect = bh / float(bw)
            if aspect < 1.0 or aspect > 5.0:
                continue
            candidates.append(contour)
        return candidates

    def _recognise_digits(self, mask: np.ndarray, contours: List[np.ndarray]) -> List[Tuple[str, float]]:
        digits: List[Tuple[str, float, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = mask[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi, (48, 64), interpolation=cv2.INTER_AREA)
            roi_resized = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY)[1]
            best_digit = None
            best_score = -1.0
            for digit, template in self.digit_templates.items():
                result = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
                score = float(result[0][0])
                if score > best_score:
                    best_score = score
                    best_digit = digit
            if best_digit is not None:
                digits.append((best_digit, best_score, x))
        digits.sort(key=lambda item: item[2])
        return [(digit, score) for digit, score, _ in digits]
