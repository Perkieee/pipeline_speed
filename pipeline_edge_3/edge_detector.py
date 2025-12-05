import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _ensure_dir(path: str) -> None:
    if not path:
        return
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class EdgeDetector:
    """
    Lightweight color/contour-based detector adapted from the provided script.

    It builds a yellow mask, cleans it with morphology, finds contours, and returns
    normalized bounding boxes (xmin, ymin, xmax, ymax, score) for the pipeline to
    crop and batch. Optional crop saving is kept for the accuracy/debug workflow.
    """

    def __init__(
        self,
        mode: str = "accuracy",
        crop_save_dir: str = "/home/sir/Kevin_Walter/tests/pipeline_edge/crops",
        lower_yellow: Tuple[int, int, int] = (20, 80, 80),
        upper_yellow: Tuple[int, int, int] = (35, 255, 255),
        kernel_size: int = 5,
        min_area_ratio: float = 0.01,
        min_area_floor: int = 500,
        margin_pixels: int = 10,
        save_crops: Optional[bool] = None,
        show_mask: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.mode = mode
        self.crop_save_dir = crop_save_dir
        self.lower_yellow = np.array(lower_yellow, dtype=np.uint8)
        self.upper_yellow = np.array(upper_yellow, dtype=np.uint8)
        self.kernel_size = int(kernel_size)
        self.min_area_ratio = float(min_area_ratio)
        self.min_area_floor = int(min_area_floor)
        self.margin_pixels = int(margin_pixels)
        self.save_crops = save_crops if save_crops is not None else mode == "accuracy"
        self.show_mask = show_mask
        self.logger = logger or logging.getLogger("EdgeDetector")

        if self.save_crops:
            _ensure_dir(self.crop_save_dir)

    def detect(self, frame_rgb: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Run the edge/contour-based detection on an RGB frame.

        Returns:
            detections: List of (xmin, ymin, xmax, ymax, score) normalized to [0, 1]
            detect_ms: Detection wall time in milliseconds
            post_ms: Placeholder (0.0) for compatibility with previous interface
            debug: Dict with optional mask and crop count for preview/logging
        """
        t0 = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_img, w_img = frame_bgr.shape[:2]
        image_area = h_img * w_img
        min_contour_area = max(int(image_area * self.min_area_ratio), self.min_area_floor)

        detections: List[Tuple[float, float, float, float, float]] = []
        crops_saved = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            x = max(0, x - self.margin_pixels)
            y = max(0, y - self.margin_pixels)
            x2 = min(w_img, x + w + 2 * self.margin_pixels)
            y2 = min(h_img, y + h + 2 * self.margin_pixels)
            if x >= x2 or y >= y2:
                continue

            xmin_n = x / w_img
            ymin_n = y / h_img
            xmax_n = x2 / w_img
            ymax_n = y2 / h_img
            score = min(1.0, area / float(image_area))
            detections.append((xmin_n, ymin_n, xmax_n, ymax_n, score))

            if self.save_crops:
                crop = frame_bgr[y:y2, x:x2]
                if crop.size > 0:
                    ts_ms = int(time.time() * 1000)
                    filename = f"edge_crop_{ts_ms}_{len(detections)}.jpg"
                    save_path = os.path.join(self.crop_save_dir, filename)
                    try:
                        cv2.imwrite(save_path, crop)
                        crops_saved += 1
                    except Exception as exc:
                        self.logger.warning("Failed to save crop to %s: %s", save_path, exc)

        detect_ms = (time.perf_counter() - t0) * 1000.0
        debug = {"mask": mask_clean if self.show_mask else None, "crops_saved": crops_saved}
        return detections, detect_ms, 0.0, debug
