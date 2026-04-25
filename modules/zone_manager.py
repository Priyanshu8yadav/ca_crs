"""
zone_manager.py
───────────────
Manages the venue graph: zones, cameras, gates, and adjacency.
Each camera corresponds to one zone (node). Gates are edges between zones.

Venue Graph:
  Camera 1 (Entry Corridor) ──Gate-1──► Camera 2 (Main Hall)
  Camera 2 (Main Hall)      ──Gate-2──► Camera 3 (Exit Plaza)
  Camera 3 (Exit Plaza)     ──Gate-3──► Outside
"""

import cv2
import numpy as np
import json


class ZoneManager:
    """
    Manages zone definitions for a venue.
    Zones can be regions within a single camera frame (single-camera mode)
    or entire camera feeds (multi-camera mode).
    """

    def __init__(
        self,
        frame_shape=None,
        config_path=None,
        mode="single",
        zone_layout="full_frame",
        full_frame_area_m2=140.0,
        full_frame_capacity=980,
        full_frame_name="Crowd Zone",
        full_frame_gates=None,
    ):
        """
        Parameters
        ----------
        frame_shape : tuple (H, W) — shape of single camera frame
        config_path : str — path to JSON zone config (optional)
        mode        : "single" or "multi"
        """
        self.mode = mode
        self.zone_layout = zone_layout
        self.full_frame_area_m2 = float(full_frame_area_m2)
        self.full_frame_capacity = int(full_frame_capacity)
        self.full_frame_name = str(full_frame_name)
        self.full_frame_gates = list(
            full_frame_gates if full_frame_gates else ["Gate-Primary", "Gate-Secondary"]
        )
        if config_path:
            self.zones = self._load_config(config_path)
        elif frame_shape is not None:
            h, w = frame_shape[:2]
            self.zones = self._default_zones(h, w)
        else:
            raise ValueError("Provide frame_shape or config_path")

        self.h = frame_shape[0] if frame_shape else None
        self.w = frame_shape[1] if frame_shape else None

        if self.h and self.w:
            self._build_masks()

        # Build adjacency map from gate connections
        self.adjacency = self._build_adjacency()

    # ── Zone loading ──────────────────────────────────────────────────────

    def _load_config(self, path):
        with open(path) as f:
            return json.load(f)["zones"]

    def _default_zones(self, h, w):
        """Default single-camera config."""
        if self.zone_layout == "full_frame":
            return [
                {
                    "id": 0,
                    "name": self.full_frame_name,
                    "polygon": [(0, 0), (w, 0), (w, h), (0, h)],
                    "area_m2": self.full_frame_area_m2,
                    "capacity": self.full_frame_capacity,
                    "gates": self.full_frame_gates,
                    "adjacent_zones": [],
                    "camera_id": 0,
                }
            ]

        if self.zone_layout != "three_strip":
            raise ValueError(f"Unsupported zone layout: {self.zone_layout}")

        # Three horizontal strips — legacy single-camera config.
        return [
            {
                "id": 0,
                "name": "Entry Zone",
                "polygon": [(0, 0), (w, 0),
                            (w, h // 3), (0, h // 3)],
                "area_m2": 40.0,
                "capacity": 280,
                "gates": ["Gate-1", "Gate-2"],
                "adjacent_zones": [1],   # connects to Central Zone
                "camera_id": 0,
            },
            {
                "id": 1,
                "name": "Central Zone",
                "polygon": [(0, h // 3), (w, h // 3),
                            (w, 2 * h // 3), (0, 2 * h // 3)],
                "area_m2": 60.0,
                "capacity": 420,
                "gates": ["Gate-2", "Gate-3"],
                "adjacent_zones": [0, 2],
                "camera_id": 0,
            },
            {
                "id": 2,
                "name": "Exit Zone",
                "polygon": [(0, 2 * h // 3), (w, 2 * h // 3),
                            (w, h), (0, h)],
                "area_m2": 40.0,
                "capacity": 280,
                "gates": ["Gate-3", "Gate-4"],
                "adjacent_zones": [1],
                "camera_id": 0,
            },
        ]

    def _build_masks(self):
        """Pre-compute binary polygon masks for each zone."""
        self.masks = []
        for zone in self.zones:
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            pts = np.array(zone["polygon"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self.masks.append(mask)

    def _build_adjacency(self):
        """Build zone adjacency dict from zone configs."""
        adj = {}
        for zone in self.zones:
            adj[zone["id"]] = zone.get("adjacent_zones", [])
        return adj

    # ── Person assignment ─────────────────────────────────────────────────

    def assign_detections_to_zones(self, boxes):
        """
        Assign detected persons (bounding boxes) to zones.

        Uses the bottom-center point of each box instead of the centroid.
        For elevated crowd footage this is a better approximation of the
        person's ground position, so counts do not drift into the wrong
        horizontal zone when only the upper body is detected.

        Parameters
        ----------
        boxes : list of (x1,y1,x2,y2)

        Returns
        -------
        dict : {zone_name: count}
        """
        zone_boxes = self.assign_boxes_to_zones(boxes)
        return {name: len(items) for name, items in zone_boxes.items()}

    def assign_boxes_to_zones(self, boxes):
        """
        Assign detected person boxes to zones and keep the box lists.
        """
        zone_boxes = {z["name"]: [] for z in self.zones}
        if not hasattr(self, "masks"):
            return zone_boxes

        for (x1, y1, x2, y2) in boxes:
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            cy = max(0, min(cy, self.h - 1))
            cx = max(0, min(cx, self.w - 1))
            for i, zone in enumerate(self.zones):
                if self.masks[i][cy, cx] > 0:
                    zone_boxes[zone["name"]].append((x1, y1, x2, y2))
                    break
        return zone_boxes

    # ── Zone property accessors ───────────────────────────────────────────

    def get_zone(self, name_or_id):
        for z in self.zones:
            if z["name"] == name_or_id or z["id"] == name_or_id:
                return z
        return None

    def get_adjacent_zones(self, zone_id):
        """Return list of adjacent zone dicts."""
        adj_ids = self.adjacency.get(zone_id, [])
        return [self.get_zone(i) for i in adj_ids]

    def get_adjacent_zone_ids(self, zone_id):
        return self.adjacency.get(zone_id, [])

    # ── Visualization ─────────────────────────────────────────────────────

    def draw_zones(self, frame, zone_risk_levels=None):
        """
        Draw zone boundaries on frame with optional risk-level coloring.

        Parameters
        ----------
        zone_risk_levels : dict {zone_name: "SAFE"|"WARNING"|"DANGER"}
        """
        COLORS = {
            "SAFE":    (0, 200, 0),
            "WARNING": (0, 165, 255),
            "DANGER":  (0, 0, 255),
            None:      (200, 200, 200),
        }
        for i, zone in enumerate(self.zones):
            risk = zone_risk_levels.get(zone["name"]) if zone_risk_levels else None
            color = COLORS.get(risk, COLORS[None])
            pts = np.array(zone["polygon"], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
            # Label at centroid
            cx = int(np.mean([p[0] for p in zone["polygon"]]))
            cy = int(np.mean([p[1] for p in zone["polygon"]]))
            cv2.putText(frame, zone["name"],
                        (cx - 45, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        color, 1, cv2.LINE_AA)
        return frame
