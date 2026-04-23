"""
motion_module.py
────────────────
Lucas-Kanade sparse optical flow for speed and
directional conflict extraction.

For handheld panic footage, raw optical flow tends to over-report
directional conflict because camera shake mixes background motion with
crowd motion. We therefore estimate a global camera transform first and
measure residual motion after compensation.
"""

import cv2
import numpy as np

V_MAX        = 15.0   # pixels/frame → normalized speed = 1.0
MIN_POINTS   = 20     # minimum tracked points for reliable estimate
MIN_MOVE_MAG = 0.8    # ignore tiny residual vectors after stabilization
CONFLICT_ANGLE = 2 * np.pi / 3  # count only strongly opposing motion as conflict


class MotionAnalyzer:

    def __init__(self):
        self.feature_params = dict(
            maxCorners=200, qualityLevel=0.2,
            minDistance=7,  blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def analyze(self, prev_gray, curr_gray):
        """
        Returns (S_norm, C_norm, flow_vectors).
        flow_vectors: np.ndarray shape (N,2) — used for per-zone analysis.
        """
        p0 = cv2.goodFeaturesToTrack(
            prev_gray, mask=None, **self.feature_params
        )
        if p0 is None or len(p0) < MIN_POINTS:
            return 0.0, 0.0, np.empty((0, 2))

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **self.lk_params
        )
        good_old = p0[st == 1]
        good_new = p1[st == 1]

        if len(good_new) < MIN_POINTS:
            return 0.0, 0.0, np.empty((0, 2))

        flow_vecs = self._compensated_flow(good_old, good_new)
        if len(flow_vecs) == 0:
            return 0.0, 0.0, np.empty((0, 2))

        magnitudes = np.linalg.norm(flow_vecs, axis=1)
        moving_mask = magnitudes >= MIN_MOVE_MAG
        if np.count_nonzero(moving_mask) < max(8, MIN_POINTS // 3):
            return 0.0, 0.0, np.empty((0, 2))

        flow_vecs = flow_vecs[moving_mask]
        snorm = self._speed(flow_vecs)
        cnorm = self._conflict(flow_vecs)
        return snorm, cnorm, flow_vecs

    # ── helpers ───────────────────────────────────────────────────────────

    def _compensated_flow(self, old_pts, new_pts):
        old_pts = np.asarray(old_pts, dtype=np.float32).reshape(-1, 1, 2)
        new_pts = np.asarray(new_pts, dtype=np.float32).reshape(-1, 1, 2)

        if len(old_pts) < 6:
            return (new_pts - old_pts).reshape(-1, 2)

        matrix, _ = cv2.estimateAffinePartial2D(
            old_pts,
            new_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )

        if matrix is None:
            return (new_pts - old_pts).reshape(-1, 2)

        predicted = cv2.transform(old_pts, matrix)
        return (new_pts - predicted).reshape(-1, 2)

    def _speed(self, vecs):
        magnitudes = np.linalg.norm(vecs, axis=1)
        if len(magnitudes) == 0:
            return 0.0

        # Use an upper percentile of stabilized motion to make panic bursts
        # visible even when many tracked points belong to static background.
        robust_speed = np.percentile(magnitudes, 75)
        return float(min(robust_speed / V_MAX, 1.0))

    def _conflict(self, vecs):
        if len(vecs) < 2:
            return 0.0
        mags = np.linalg.norm(vecs, axis=1)
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        mean_sin = np.average(np.sin(angles), weights=np.maximum(mags, 1e-3))
        mean_cos = np.average(np.cos(angles), weights=np.maximum(mags, 1e-3))
        dom      = np.arctan2(mean_sin, mean_cos)
        diff     = np.abs(angles - dom)
        diff     = np.where(diff > np.pi, 2 * np.pi - diff, diff)
        opposing = diff > CONFLICT_ANGLE
        return float(np.average(opposing.astype(np.float32),
                                weights=np.maximum(mags, 1e-3)))
