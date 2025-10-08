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
    """Template-based OCR for circular speed limit signs."""

    def __init__(
        self,
        candidate_values: Iterable[int] = (10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120),
        template_size: Tuple[int, int] = (120, 120),
        match_threshold: float = 0.35,
    ) -> None:
        self.template_size = template_size
        self.match_threshold = match_threshold
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 2.6
        self.thickness = 6
        self.templates = self._build_templates(candidate_values)

    def infer(self, crop_rgb: np.ndarray) -> Optional[Tuple[str, float]]:
        if crop_rgb.size == 0:
            return None

        crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 0)

        h, w = crop_gray.shape
        if h < 20 or w < 20:
            return None

        mask = np.zeros_like(crop_gray)
        radius = int(min(h, w) * 0.45)
        cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
        masked = cv2.bitwise_and(crop_gray, crop_gray, mask=mask)

        _, digits = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        digits = cv2.medianBlur(digits, 5)
        digits = cv2.dilate(digits, np.ones((3, 3), np.uint8), iterations=1)

        cx, cy = w // 2, h // 2
        side = int(min(h, w) * 0.80)
        half = side // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        core = digits[y1:y2, x1:x2]
        if core.size == 0:
            core = digits

        resized = cv2.resize(core, self.template_size, interpolation=cv2.INTER_AREA)
        resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY)[1]

        best_value = None
        best_score = -1.0
        for value, template in self.templates.items():
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_score = score
                best_value = value

        if best_value is None or best_score < self.match_threshold:
            return None
        return best_value, best_score

    def _build_templates(self, candidate_values: Iterable[int]) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        width, height = self.template_size
        for value in candidate_values:
            text = str(value)
            canvas = np.zeros((height, width), dtype=np.uint8)
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
            text_x = max(0, (width - text_size[0]) // 2)
            text_y = max(text_size[1] + (height - text_size[1]) // 2 - 5, text_size[1])
            cv2.putText(canvas, text, (text_x, text_y), self.font, self.font_scale, 255, self.thickness, cv2.LINE_AA)
            template = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)[1]
            templates[text] = template
        return templates
