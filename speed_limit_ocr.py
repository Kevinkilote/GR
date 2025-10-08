"""Digit recognizer for speed-limit traffic signs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class OCRResult:
    value: str
    score: float


class SpeedLimitOCR:
    """Template-based OCR tailored for circular speed limit signs."""

    def __init__(
        self,
        candidate_digits: Iterable[int] = range(0, 10),
        digit_size: Tuple[int, int] = (28, 42),
        match_threshold: float = 0.35,
    ) -> None:
        self.digit_size = digit_size
        self.match_threshold = match_threshold
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.5
        self.thickness = 3
        self.digit_templates = self._build_digit_templates(candidate_digits)

    def infer(self, crop_rgb: np.ndarray) -> Optional[Tuple[str, float]]:
        if crop_rgb.size == 0:
            return None

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape
        if h < 24 or w < 24:
            return None

        # Emphasise the inner region of the sign (remove red border)
        mask = np.zeros_like(gray)
        radius = int(min(h, w) * 0.45)
        cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
        inner = cv2.bitwise_and(gray, gray, mask=mask)

        # High-contrast threshold for digits (digits are dark)
        _, thresh = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.medianBlur(thresh, 5)
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_regions = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw <= 0 or ch <= 0:
                continue
            if ch < h * 0.35 or cw < w * 0.12:
                continue
            if ch > h * 0.95 or cw > w * 0.75:
                continue
            cx = x + cw / 2
            if cx < w * 0.2 or cx > w * 0.8:
                continue
            region = thresh[y : y + ch, x : x + cw]
            digit_regions.append((x, region))

        if not digit_regions:
            return None

        digit_regions.sort(key=lambda item: item[0])
        digits: list[str] = []
        scores: list[float] = []
        for _, region in digit_regions[:3]:
            roi = self._prepare_digit(region)
            digit, score = self._match_digit(roi)
            if digit is None or score < self.match_threshold:
                return None
            digits.append(digit)
            scores.append(score)

        if not digits:
            return None
        value = ''.join(digits)
        score = float(np.mean(scores)) if scores else 0.0
        return value, score

    def _prepare_digit(self, region: np.ndarray) -> np.ndarray:
        region = cv2.resize(region, self.digit_size, interpolation=cv2.INTER_AREA)
        region = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY)[1]
        return region

    def _build_digit_templates(self, candidate_digits: Iterable[int]) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        width, height = self.digit_size
        for digit in candidate_digits:
            canvas = np.zeros((height, width), dtype=np.uint8)
            text = str(digit)
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
            text_x = max(0, (width - text_size[0]) // 2)
            text_y = max(text_size[1] + (height - text_size[1]) // 2 - 4, text_size[1])
            cv2.putText(canvas, text, (text_x, text_y), self.font, self.font_scale, 255, self.thickness, cv2.LINE_AA)
            template = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)[1]
            templates[text] = template
        return templates

    def _match_digit(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        best_digit = None
        best_score = -1.0
        for digit, template in self.digit_templates.items():
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_digit = digit
                best_score = score
        return best_digit, best_score
