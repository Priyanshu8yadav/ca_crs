"""
orchestrator.py
───────────────
Multi-camera central orchestrator.

Implements the "Venue Graph" concept:
  - Each camera = one zone (node)
  - Gates = edges between zones
  - Central GRS = weighted aggregate of all zone CA-CRS scores

Triage: When multiple zones hit DANGER simultaneously,
zones are sorted by absolute CA-CRS score (descending)
so resources go to the highest-risk zone first.

Edge processing: Each camera feed is processed independently
(in parallel via threading). Only lightweight risk scalars
and causal vectors are transmitted to this orchestrator —
not raw video streams. This enables decentralized edge
deployment where each camera node runs CA-CRS+ locally.
"""

import threading
import time
import queue


class VenueOrchestrator:
    """
    Central orchestrator for multi-camera CA-CRS+ deployment.
    Aggregates per-zone risk scalars into a Global Risk Score (GRS).
    """

    def __init__(self, zone_ids: list):
        """
        Parameters
        ----------
        zone_ids : list of zone ID integers
        """
        self.zone_ids   = zone_ids
        self.zone_data  = {zid: {} for zid in zone_ids}
        self.lock       = threading.Lock()
        self._update_q  = queue.Queue()

    # ── Data ingestion (called by each camera node) ───────────────────────

    def update_zone(self, zone_id: int, data: dict):
        """
        Receive a risk report from one camera node.

        data should contain:
          crs          : float
          risk_level   : str
          factor       : str
          gate_actions : list
          count        : int
          timestamp    : float
        """
        with self.lock:
            self.zone_data[zone_id] = data

    # ── Global Risk Score ─────────────────────────────────────────────────

    def compute_grs(self) -> dict:
        """
        Compute Global Risk Score as weighted mean of zone CA-CRS scores.
        Zones with DANGER get double weight (urgency weighting).

        Returns dict with GRS, worst zone, triage order.
        """
        with self.lock:
            data = dict(self.zone_data)

        if not data or all(not v for v in data.values()):
            return {"grs": 0.0, "worst_zone": None, "triage_order": []}

        total_weight = 0.0
        weighted_sum = 0.0
        zone_scores  = []

        for zid, zdata in data.items():
            if not zdata:
                continue
            crs   = zdata.get("crs", 0.0)
            level = zdata.get("risk_level", "SAFE")
            weight = 2.0 if level == "DANGER" else 1.0
            weighted_sum += crs * weight
            total_weight += weight
            zone_scores.append((zid, crs, level))

        grs = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Triage: sort by CRS descending
        zone_scores.sort(key=lambda x: x[1], reverse=True)
        worst_zone    = zone_scores[0][0] if zone_scores else None
        triage_order  = [z[0] for z in zone_scores]

        # Adjacent zone CRS map (for ripple check in gate recommender)
        adjacent_scores = {}
        for zid, crs, _ in zone_scores:
            adjacent_scores[zid] = crs

        return {
            "grs":             round(grs, 4),
            "worst_zone":      worst_zone,
            "triage_order":    triage_order,
            "zone_scores":     {z[0]: z[1] for z in zone_scores},
            "adjacent_scores": adjacent_scores,
        }

    def get_zone_crs(self, zone_id: int) -> float:
        with self.lock:
            return self.zone_data.get(zone_id, {}).get("crs", 0.0)

    def get_adjacent_crs(self, zone_id: int,
                         adjacent_ids: list) -> dict:
        """Return {adj_id: crs} for all adjacent zones."""
        result = {}
        with self.lock:
            for aid in adjacent_ids:
                result[aid] = self.zone_data.get(
                    aid, {}
                ).get("crs", 0.0)
        return result

    def all_zone_data(self) -> dict:
        with self.lock:
            return dict(self.zone_data)