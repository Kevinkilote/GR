"""Digit recognizer for speed-limit traffic signs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class OCRResult:
    value: str
    score: float


class SpeedLimitOCR:
    """Heuristic OCR that extracts the numeric value from circular limit signs."""

    def __init__(self, template_size: Tuple[int, int] = (40, 60)) -> None:
        self.template_size = template_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.4
        self.thickness = 3
        self.match_threshold = 0.5
        self.templates: Dict[str, np.ndarray] = self._build_templates()

    def infer(self, crop_rgb: np.ndarray) -> Optional[Tuple[str, float]]:
        if crop_rgb.size == 0:
            return None
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = thresh.shape
        mask = np.zeros_like(thresh)
        radius = int(min(h, w) * 0.45)
        cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
        digits_mask = cv2.bitwise_and(thresh, mask)

        contours, _ = cv2.findContours(digits_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < h * 0.3 or cw < w * 0.12:
                continue
            if ch > h * 0.95 or cw > w * 0.8:
                continue
            roi = digits_mask[y : y + ch, x : x + cw]
            candidates.append((x, roi))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        digits: list[str] = []
        scores: list[float] = []
        for _, roi in candidates[:3]:
            resized = cv2.resize(roi, self.template_size, interpolation=cv2.INTER_AREA)
            digit, score = self._match_digit(resized)
            if digit is None or score < self.match_threshold:
                return None
            digits.append(digit)
            scores.append(float(score))

        value = ''.join(digits)
        if not value:
            return None
        return value, float(np.mean(scores))

    def _build_templates(self) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        width, height = self.template_size
        for digit in range(10):
            canvas = np.zeros((height, width), dtype=np.uint8)
            text = str(digit)
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
            text_x = max(0, (width - text_size[0]) // 2)
            text_y = max(text_size[1] + (height - text_size[1]) // 2 - 5, text_size[1])
            cv2.putText(canvas, text, (text_x, text_y), self.font, self.font_scale, 255, self.thickness, cv2.LINE_AA)
            _, binary = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
            templates[text] = binary
        return templates

    def _match_digit(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        best_digit = None
        best_score = -1.0
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        for digit, template in self.templates.items():
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_digit = digit
                best_score = score
        return best_digit, best_score
