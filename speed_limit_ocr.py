"""Template-based OCR for circular speed-limit signs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

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
        candidate_values: Sequence[int] = (
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            100,
            110,
            120,
            130,
        ),
        template_size: Tuple[int, int] = (140, 140),
        match_threshold: float = 0.35,
    ) -> None:
        self.template_size = template_size
        self.match_threshold = match_threshold
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 2.8
        self.thickness = 6
        self.templates: Dict[str, np.ndarray] = self._build_templates(candidate_values)

    def infer(self, crop_rgb: np.ndarray) -> Optional[OCRResult]:
        if crop_rgb.size == 0:
            return None

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape
        if h < 32 or w < 32:
            return None

        mask = np.zeros_like(gray)
        radius = int(min(h, w) * 0.45)
        cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
        masked = cv2.bitwise_and(gray, gray, mask=mask)

        norm = cv2.normalize(masked, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.medianBlur(binary, 5)
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)

        ys, xs = np.where(binary > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        digit_roi = binary[y1:y2, x1:x2]
        if digit_roi.size == 0:
            digit_roi = binary

        resized = cv2.resize(digit_roi, self.template_size, interpolation=cv2.INTER_AREA)
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
        return OCRResult(value=best_value, score=best_score)

    def _build_templates(self, candidate_values: Iterable[int]) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        width, height = self.template_size
        for value in candidate_values:
            text = str(value)
            canvas = np.zeros((height, width), dtype=np.uint8)
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
            text_x = max(0, (width - text_size[0]) // 2)
            text_y = max(text_size[1] + (height - text_size[1]) // 2 - 6, text_size[1])
            cv2.putText(canvas, text, (text_x, text_y), self.font, self.font_scale, 255, self.thickness, cv2.LINE_AA)
            template = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)[1]
            templates[text] = template
        return templates
