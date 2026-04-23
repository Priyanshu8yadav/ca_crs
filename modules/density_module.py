"""
density_module.py - small-person crowd detection for overhead video.

Plain full-frame YOLO is weak when people are only a few pixels tall. This
module therefore uses fixed-size overlapping slices, runs YOLO on each slice at
high resolution, maps detections back to the full frame, and then applies
confidence-ranked NMS.

For paper results, report both:
    1. raw_detected_count: visible people detected by the model
    2. count_correction: multiplier used to estimate total crowd count

If YOLO only detects 10 visible people in a dense crowd, the right research
answer is not to pretend YOLO saw everyone. Use manual calibration or present
that run as a sensitivity/stress-test scenario.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


Box = Tuple[int, int, int, int]

# Critical density threshold in persons per square meter.
RHO_MAX = 7.0

# Conservative occlusion correction. Keep the legacy name for compatibility.
DEFAULT_COUNT_CORRECTION = 1.8
OCCLUSION_FACTOR = DEFAULT_COUNT_CORRECTION
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float


class DensityEstimator:
    def __init__(
        self,
        conf: float = 0.04,
        tile_rows: int = 5,
        tile_cols: int = 4,
        overlap: float = 0.30,
        imgsz: int = 640,
        model_name: str = "yolov8m.pt",
        iou_thresh: float = 0.45,
        count_correction: float = DEFAULT_COUNT_CORRECTION,
        max_det_per_tile: int = 3000,
        tile_size: int = 320,
        full_frame_pass: bool = False,
        augment: bool = False,
        min_box_size: int = 2,
        max_box_area_frac: float = 0.20,
    ):
        """
        Parameters
        ----------
        conf : float
            Low confidence threshold for tiny overhead people.
        tile_rows, tile_cols : int
            Fallback grid if tile_size <= 0.
        overlap : float
            Fractional overlap between adjacent tiles.
        imgsz : int
            YOLO inference size per tile. 640 is a Mac-friendly default.
        model_name : str
            YOLO weights path/name. Use "auto" to prefer local medium models.
        iou_thresh : float
            NMS IoU threshold for duplicate detections across tiles.
        count_correction : float
            Multiplier from raw detections to estimated total crowd count.
        max_det_per_tile : int
            Maximum detections returned per tile.
        tile_size : int
            Fixed tile side length in original pixels. 256-384 is best when
            people are only 4-12 pixels tall. Set <=0 to use tile_rows/cols.
        full_frame_pass : bool
            Also run YOLO once on the full frame to catch nearby large people.
        augment : bool
            Ultralytics test-time augmentation. Slower, sometimes better.
        min_box_size : int
            Minimum width/height in original pixels.
        max_box_area_frac : float
            Discard boxes covering an implausibly large fraction of the frame.
        """
        self.conf = float(conf)
        self.tile_rows = max(1, int(tile_rows))
        self.tile_cols = max(1, int(tile_cols))
        self.overlap = max(0.0, min(float(overlap), 0.90))
        self.imgsz = int(imgsz)
        self.model_name = model_name
        self.iou_thresh = float(iou_thresh)
        self.count_correction = max(0.0, float(count_correction))
        self.max_det_per_tile = max(1, int(max_det_per_tile))
        self.tile_size = int(tile_size)
        self.full_frame_pass = bool(full_frame_pass)
        self.augment = bool(augment)
        self.min_box_size = max(1, int(min_box_size))
        self.max_box_area_frac = max(0.0, min(float(max_box_area_frac), 1.0))
        self.model = self._load_model()

    # Model loading -----------------------------------------------------

    def _resolve_candidate_path(self, candidate: str) -> Path:
        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return path
        if path.exists():
            return path.resolve()
        project_local = PROJECT_ROOT / candidate
        if project_local.exists():
            return project_local
        return path

    def _model_candidates(self) -> List[str]:
        if self.model_name != "auto":
            return [self.model_name]

        # Prefer local medium/small weights first so "auto" stays usable on
        # laptops and does not immediately jump to a very slow large model.
        preferred = [
            "yolov8m.pt",
            "yolov8s.pt",
            "yolov8n.pt",
            "yolo11x.pt",
            "yolo11l.pt",
            "yolov8x.pt",
        ]
        local = [
            name for name in preferred
            if self._resolve_candidate_path(name).exists()
        ]
        return local or preferred

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[DensityEstimator] ultralytics not found; using stub.")
            return None

        last_error = None
        for candidate in self._model_candidates():
            try:
                candidate_path = self._resolve_candidate_path(candidate)
                load_target = str(candidate_path) if candidate_path.exists() else candidate
                model = YOLO(load_target)
                print(
                    "[DensityEstimator] Loaded "
                    f"{candidate_path.name if candidate_path.exists() else candidate} "
                    f"| conf={self.conf} | imgsz={self.imgsz} "
                    f"| tile_size={self.tile_size} | grid="
                    f"{self.tile_rows}x{self.tile_cols} "
                    f"| correction={self.count_correction}"
                )
                self.loaded_model_name = (
                    candidate_path.name if candidate_path.exists() else candidate
                )
                return model
            except Exception as exc:
                last_error = exc
                print(f"[DensityEstimator] Could not load {candidate}: {exc}")

        print(f"[DensityEstimator] All YOLO loads failed: {last_error}")
        print("[DensityEstimator] Falling back to stub detections.")
        return None

    # Public API --------------------------------------------------------

    def detect(self, frame, corrected: bool = True):
        """
        Detect people in a frame.

        Returns
        -------
        count : int
            Estimated count if corrected=True, otherwise raw YOLO count.
        boxes : list[tuple]
            De-duplicated full-frame boxes in (x1, y1, x2, y2) format.
        """
        detections = self.detect_detailed(frame)
        raw_count = len(detections)
        count = self.correct_count(raw_count) if corrected else raw_count
        boxes = [det.box for det in detections]
        return count, boxes

    def detect_detailed(self, frame) -> List[Detection]:
        """Return de-duplicated detections with confidence scores."""
        if self.model is None:
            return [
                Detection(box=box, score=0.50)
                for box in self._stub_boxes(frame)
            ]

        h, w = frame.shape[:2]
        boxes: List[Box] = []
        scores: List[float] = []

        if self.full_frame_pass:
            self._append_tile_detections(
                frame, 0, 0, w, h, boxes, scores, frame_w=w, frame_h=h
            )

        for x1, y1, x2, y2 in self._tile_coords(h, w):
            self._append_tile_detections(
                frame, x1, y1, x2, y2, boxes, scores, frame_w=w, frame_h=h
            )

        keep = self._nms_indices(boxes, scores, self.iou_thresh)
        return [Detection(box=boxes[i], score=scores[i]) for i in keep]

    def correct_count(self, raw_count: int) -> int:
        """Convert raw visible detections to estimated total crowd count."""
        return int(round(max(0, int(raw_count)) * self.count_correction))

    def density(
        self,
        count: int,
        area_m2: float,
        count_is_raw: bool = False,
    ) -> float:
        """Density in persons per square meter."""
        estimate = self.correct_count(count) if count_is_raw else count
        return float(estimate) / max(float(area_m2), 1.0)

    def dnorm(
        self,
        count: int,
        area_m2: float,
        count_is_raw: bool = False,
    ) -> float:
        """Normalized density in [0, 1]."""
        return min(self.density(count, area_m2, count_is_raw) / RHO_MAX, 1.0)

    def risk_from_density(self, count: int, area_m2: float) -> str:
        """Simple density-only label useful for debugging/calibration."""
        rho = self.density(count, area_m2)
        if rho >= RHO_MAX:
            return "DANGER"
        if rho >= 0.5 * RHO_MAX:
            return "WARNING"
        return "SAFE"

    # Inference ---------------------------------------------------------

    def _append_tile_detections(
        self,
        frame,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        boxes: List[Box],
        scores: List[float],
        frame_w: int,
        frame_h: int,
    ) -> None:
        tile = frame[y1:y2, x1:x2]
        if tile.size == 0:
            return

        results = self.model(
            tile,
            classes=[0],
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=self.max_det_per_tile,
            augment=self.augment,
            verbose=False,
        )

        frame_area = max(1, frame_w * frame_h)
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()

            for box, score in zip(xyxy, confs):
                bx1 = int(round(box[0] + x1))
                by1 = int(round(box[1] + y1))
                bx2 = int(round(box[2] + x1))
                by2 = int(round(box[3] + y1))

                clipped = self._clip_box(
                    (bx1, by1, bx2, by2), frame_w, frame_h
                )
                if clipped is None:
                    continue

                if not self._is_plausible_person_box(clipped, frame_area):
                    continue

                boxes.append(clipped)
                scores.append(float(score))

    def _clip_box(self, box: Box, frame_w: int, frame_h: int):
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), frame_w - 1))
        y1 = max(0, min(int(y1), frame_h - 1))
        x2 = max(0, min(int(x2), frame_w))
        y2 = max(0, min(int(y2), frame_h))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _is_plausible_person_box(self, box: Box, frame_area: int) -> bool:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        if bw < self.min_box_size or bh < self.min_box_size:
            return False
        if (bw * bh) / frame_area > self.max_box_area_frac:
            return False
        return True

    # Tiling ------------------------------------------------------------

    def _tile_coords(self, h: int, w: int) -> List[Box]:
        """Generate overlapping tile coordinates."""
        if self.tile_size > 0:
            return self._fixed_tile_coords(h, w)
        return self._grid_tile_coords(h, w)

    def _fixed_tile_coords(self, h: int, w: int) -> List[Box]:
        size = max(32, int(self.tile_size))
        step = max(1, int(size * (1.0 - self.overlap)))

        def starts(length: int) -> List[int]:
            if length <= size:
                return [0]
            vals = list(range(0, length - size + 1, step))
            last = length - size
            if vals[-1] != last:
                vals.append(last)
            return vals

        tiles: List[Box] = []
        for y in starts(h):
            for x in starts(w):
                tiles.append((x, y, min(x + size, w), min(y + size, h)))
        return tiles

    def _grid_tile_coords(self, h: int, w: int) -> List[Box]:
        tiles: List[Box] = []
        row_h = h / self.tile_rows
        col_w = w / self.tile_cols
        pad_y = int(row_h * self.overlap)
        pad_x = int(col_w * self.overlap)

        for r in range(self.tile_rows):
            for c in range(self.tile_cols):
                y1 = max(0, int(r * row_h) - pad_y)
                y2 = min(h, int((r + 1) * row_h) + pad_y)
                x1 = max(0, int(c * col_w) - pad_x)
                x2 = min(w, int((c + 1) * col_w) + pad_x)
                if x2 > x1 and y2 > y1:
                    tiles.append((x1, y1, x2, y2))
        return tiles

    # NMS ---------------------------------------------------------------

    def _nms_indices(
        self,
        boxes: Sequence[Box],
        scores: Sequence[float],
        iou_thresh: float,
    ) -> List[int]:
        """Confidence-ranked non-maximum suppression."""
        if not boxes:
            return []

        boxes_arr = np.array(boxes, dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)

        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]

        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores_arr.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            rest = order[1:]
            if rest.size == 0:
                break

            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])

            iw = np.maximum(0.0, ix2 - ix1)
            ih = np.maximum(0.0, iy2 - iy1)
            inter = iw * ih

            denom = areas[i] + areas[rest] - inter + 1e-6
            iou = inter / denom
            order = rest[np.where(iou <= iou_thresh)[0]]

        return keep

    # Stub fallback -----------------------------------------------------

    def _stub_boxes(self, frame) -> List[Box]:
        import random

        h, w = frame.shape[:2]
        n = random.randint(15, 80)
        boxes: List[Box] = []
        for _ in range(n):
            bw = random.randint(12, 36)
            bh = random.randint(18, 54)
            bx = random.randint(0, max(1, w - bw))
            by = random.randint(0, max(1, h - bh))
            boxes.append((bx, by, bx + bw, by + bh))
        return boxes


def model_exists_locally(model_name: str) -> bool:
    """Convenience helper for scripts that want to avoid downloads."""
    path = Path(model_name)
    return path.exists() or (PROJECT_ROOT / model_name).exists()
