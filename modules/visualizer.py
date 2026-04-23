"""
visualizer.py
─────────────
Draws the complete CA-CRS+ dashboard on each video frame.
"""

import cv2
import numpy as np
from modules.risk_scoring import SAFE, WARNING, DANGER

RISK_COLORS = {
    SAFE:    (0, 200, 0),
    WARNING: (0, 165, 255),
    DANGER:  (0, 0, 255),
}
RES_COLORS = {
    "ADEQUATE": (0, 200, 0),
    "STRAINED": (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}
GATE_COLORS = {
    "OPEN":     (0, 200, 0),
    "CLOSE":    (0, 0, 255),
    "REDIRECT": (255, 140, 0),
    "HOLD":     (160, 160, 160),
}


class Visualizer:

    def draw_single_zone(self, frame, boxes, zone_results,
                         resource_status, grs_data=None):
        """Draw full dashboard for single-camera multi-zone view."""
        out = frame.copy()
        h, w = out.shape[:2]
        panel_w = 295

        # Draw bounding boxes
        for zname, zr in zone_results.items():
            color = RISK_COLORS.get(zr.get("risk", SAFE))
            for box in zr.get("boxes", []):
                x1, y1, x2, y2 = box
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

        # Semi-transparent sidebar
        overlay = out.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, h),
                      (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)

        x, y = w - panel_w + 8, 18

        def txt(text, yy, color=(210, 210, 210),
                scale=0.40, bold=1):
            cv2.putText(out, text, (x, yy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, bold, cv2.LINE_AA)

        # Header
        txt("CA-CRS+ SYSTEM", y, (255, 255, 255), 0.52, 2)
        y += 16
        txt("Multi-Zone Crowd Safety", y, (140, 140, 140), 0.38)
        y += 14

        # GRS
        if grs_data:
            grs = grs_data.get("grs", 0.0)
            grs_color = RISK_COLORS[DANGER] if grs >= 0.70 else \
                        RISK_COLORS[WARNING] if grs >= 0.35 else \
                        RISK_COLORS[SAFE]
            txt(f"GRS: {grs:.3f}", y, grs_color, 0.48, 2)
            y += 16

        # Divider
        cv2.line(out, (x, y), (w - 8, y), (60, 60, 60), 1)
        y += 8

        # Per-zone blocks
        for zname, zr in zone_results.items():
            risk  = zr.get("risk", SAFE)
            crs   = zr.get("crs", 0.0)
            color = RISK_COLORS[risk]

            txt(zname, y, color, 0.43, 2); y += 13
            txt(f"  CRS:{crs:.3f}  [{risk}]", y, color); y += 12

            factor = zr.get("factor", "---")
            txt(f"  Cause: {factor}", y, (240, 200, 80)); y += 11

            for ga in zr.get("gates", []):
                gc = GATE_COLORS.get(ga["action"], (160, 160, 160))
                ripple = " [!RIPPLE]" if ga.get("ripple_blocked") else ""
                txt(f"  {ga['gate']}: {ga['action']}"
                    f"[{ga['priority']}]{ripple}",
                    y, gc); y += 11

            # Mini CRS bar
            bar_x, bar_y = x + 2, y
            bar_w = panel_w - 20
            cv2.rectangle(out, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + 5),
                          (50, 50, 50), -1)
            cv2.rectangle(out, (bar_x, bar_y),
                          (bar_x + int(crs * bar_w), bar_y + 5),
                          color, -1)
            y += 10

            cv2.line(out, (x, y), (w - 8, y), (45, 45, 45), 1)
            y += 6

        # Resource status
        rs  = resource_status.get("status", "ADEQUATE")
        rsc = RES_COLORS.get(rs, (200, 200, 200))
        txt(f"RESOURCES: {rs}", y, rsc, 0.43, 2); y += 13

        if resource_status.get("shortfall"):
            for res, amt in resource_status["shortfall"].items():
                txt(f"  NEED +{int(amt)} {res}",
                    y, (0, 0, 255)); y += 11

        triage = resource_status.get("triage_order", [])
        if triage:
            txt(f"  Triage: {' > '.join(str(z) for z in triage[:3])}",
                y, (200, 200, 100)); y += 11

        return out

    def draw_architecture_text(self, frame):
        """Overlay pipeline label at bottom of frame."""
        h, w = frame.shape[:2]
        label = "CA-CRS+ v1.0 | Zone-Aware | Gate-Specific | Resource-Aware"
        cv2.rectangle(frame, (0, h - 22), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, label, (8, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 180, 180), 1, cv2.LINE_AA)
        return frame