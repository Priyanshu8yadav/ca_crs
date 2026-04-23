"""
gate_recommender.py
───────────────────
Maps (zone, dominant_factor, risk_level) → specific gate commands.

NOVEL CONTRIBUTION: In multi-camera deployments, before issuing
a REDIRECT command, the system checks CA-CRS of adjacent zones
to prevent creating a secondary bottleneck — the "ripple effect."

Gate Actions:
  OPEN     — open gate to increase outflow
  CLOSE    — close gate to restrict inflow
  REDIRECT — direct crowd to an alternate gate
  HOLD     — maintain current state
"""

from modules.risk_scoring import (
    SAFE, WARNING, DANGER,
    FACTOR_DENSITY, FACTOR_SPEED, FACTOR_CONFLICT, FACTOR_MIXED
)

# ── Cause-to-gate action mapping ──────────────────────────────────────────
CAUSE_GATE_MAP = {
    FACTOR_DENSITY:  ("OPEN",     "OPEN"),
    FACTOR_SPEED:    ("CLOSE",    "HOLD"),
    FACTOR_CONFLICT: ("REDIRECT", "OPEN"),
    FACTOR_MIXED:    ("HOLD",     "HOLD"),
}

# ── Gate action colors for visualization (BGR) ───────────────────────────
GATE_COLORS = {
    "OPEN":     (0, 200, 0),
    "CLOSE":    (0, 0, 255),
    "REDIRECT": (0, 165, 255),
    "HOLD":     (180, 180, 180),
}

RISK_COLORS = {
    SAFE:    (0, 200, 0),
    WARNING: (0, 165, 255),
    DANGER:  (0, 0, 255),
}


class GateRecommender:

    def recommend_gates(self,
                        zone_name: str,
                        zone_id: int,
                        dominant_factor: str,
                        risk_level: str,
                        zone_gates: list,
                        adjacent_zone_scores: dict = None) -> list:
        """
        Parameters
        ----------
        zone_name            : str  — name of this zone
        zone_id              : int  — ID of this zone
        dominant_factor      : str  — DENSITY / SPEED / CONFLICT / MIXED
        risk_level           : str  — SAFE / WARNING / DANGER
        zone_gates           : list — gate names for this zone
        adjacent_zone_scores : dict — {zone_id: crs_score} for neighbours
                                      Used to check ripple effect.

        Returns
        -------
        list of gate action dicts
        """
        if risk_level == SAFE or not zone_gates:
            return [{"gate": g, "action": "HOLD",
                     "priority": "LOW", "zone": zone_name,
                     "cause": dominant_factor}
                    for g in zone_gates]

        primary_action, secondary_action = CAUSE_GATE_MAP.get(
            dominant_factor, ("HOLD", "HOLD")
        )

        # ── RIPPLE EFFECT CHECK (multi-camera novelty) ────────────────────
        # If we plan to REDIRECT, check adjacent zones.
        # If an adjacent zone also has high risk (>0.50), downgrade
        # REDIRECT to HOLD and escalate to supervisor instead.
        redirect_blocked = False
        if primary_action == "REDIRECT" and adjacent_zone_scores:
            for adj_id, adj_crs in adjacent_zone_scores.items():
                if adj_crs > 0.50:
                    redirect_blocked = True
                    break

        if redirect_blocked:
            primary_action   = "HOLD"
            secondary_action = "HOLD"
            # Flag for supervisor escalation
            ripple_warning = True
        else:
            ripple_warning = False

        priority = "HIGH" if risk_level == DANGER else "MEDIUM"
        gate_actions = []

        for i, gate in enumerate(zone_gates):
            action = primary_action if i == 0 else secondary_action
            p      = priority if i == 0 else (
                "MEDIUM" if priority == "HIGH" else "LOW"
            )
            gate_actions.append({
                "gate":           gate,
                "action":         action,
                "priority":       p,
                "zone":           zone_name,
                "zone_id":        zone_id,
                "cause":          dominant_factor,
                "ripple_blocked": ripple_warning if i == 0 else False,
            })

        return gate_actions

    def format_summary(self, gate_actions: list) -> str:
        if not gate_actions:
            return "No gate actions"
        parts = []
        for ga in gate_actions:
            rb = " [RIPPLE!]" if ga.get("ripple_blocked") else ""
            parts.append(f"{ga['gate']}:{ga['action']}[{ga['priority']}]{rb}")
        return " | ".join(parts)