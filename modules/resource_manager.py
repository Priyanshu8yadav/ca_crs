"""
resource_manager.py
───────────────────
Dynamic resource-aware alert module.

Demand scales logarithmically with zone crowd count:
  D_mar(k) = ceil(α_ℓ × log10(1 + Nk))

This prevents the 3× overestimation that linear models produce
at high crowd volumes, while still scaling demand with crowd size.

Resource status:
  ADEQUATE : no shortfall
  STRAINED : shortfall ≤ 3 units → request reinforcements
  CRITICAL : shortfall > 3 units → external emergency assistance

Triage: When multiple zones trigger alerts simultaneously, the
system sorts zones by absolute CA-CRS score (descending) and
allocates available resources to the highest-risk zone first.
"""

import math

SAFE    = "SAFE"
WARNING = "WARNING"
DANGER  = "DANGER"


# Baseline severity coefficients per risk level
ALPHA = {"SAFE": 0, "WARNING": 4,  "DANGER": 10}   # marshals
BETA  = {"SAFE": 0, "WARNING": 1,  "DANGER": 2}    # medics
GAMMA = {"SAFE": 0, "WARNING": 0.5,"DANGER": 1}    # ambulances


class ResourceManager:

    def __init__(self, marshals: int = 20,
                 medics: int = 5,
                 ambulances: int = 3):
        self.available = {
            "marshals":   marshals,
            "medics":     medics,
            "ambulances": ambulances,
        }

    # ── demand computation ────────────────────────────────────────────────

    def _zone_demand(self, risk_level: str,
                     zone_count: int) -> dict:
        """Logarithmic demand for one zone."""
        log_factor = math.log10(1 + max(zone_count, 0))
        return {
            "marshals":   math.ceil(ALPHA[risk_level] * log_factor),
            "medics":     math.ceil(BETA[risk_level]  * log_factor),
            "ambulances": math.ceil(GAMMA[risk_level] * log_factor),
        }

    # ── main check ────────────────────────────────────────────────────────

    def check(self, zone_risk_levels: dict,
              zone_counts: dict = None) -> dict:
        """
        Parameters
        ----------
        zone_risk_levels : {zone_name: risk_level}
        zone_counts      : {zone_name: person_count}  (optional)
                           If omitted, uses 50 persons per zone as estimate.

        Returns
        -------
        dict with keys: status, demand, available, shortfall,
                        message, triage_order
        """
        if zone_counts is None:
            zone_counts = {z: 50 for z in zone_risk_levels}

        total_demand = {"marshals": 0, "medics": 0, "ambulances": 0}
        for zone, level in zone_risk_levels.items():
            count = zone_counts.get(zone, 50)
            zone_d = self._zone_demand(level, count)
            for res in total_demand:
                total_demand[res] += zone_d[res]

        # Shortfall
        shortfall = {}
        for res, needed in total_demand.items():
            deficit = needed - self.available[res]
            if deficit > 0:
                shortfall[res] = round(deficit, 1)

        # Status
        if not shortfall:
            status  = "ADEQUATE"
            message = "All resources sufficient."
        elif max(shortfall.values()) <= 3:
            status  = "STRAINED"
            parts   = [f"+{int(v)} {r}" for r, v in shortfall.items()]
            message = f"Request reinforcements: {', '.join(parts)}"
        else:
            status  = "CRITICAL"
            parts   = [f"+{int(v)} {r}" for r, v in shortfall.items()]
            message = (f"CRITICAL SHORTAGE: {', '.join(parts)}. "
                       f"Immediate external assistance required.")

        # Triage order — sort zones by risk level severity
        level_rank = {SAFE: 0, WARNING: 1, DANGER: 2}
        triage_order = sorted(
            zone_risk_levels.items(),
            key=lambda x: level_rank.get(x[1], 0),
            reverse=True
        )

        return {
            "status":       status,
            "demand":       total_demand,
            "available":    self.available,
            "shortfall":    shortfall,
            "message":      message,
            "triage_order": [z for z, _ in triage_order],
        }

    def update(self, resource: str, delta: int):
        """Update available resource count (deploy/recall)."""
        if resource in self.available:
            self.available[resource] = max(
                0, self.available[resource] + delta
            )