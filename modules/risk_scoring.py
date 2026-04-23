"""
risk_scoring.py
───────────────
Non-linear CA-CRS+ formula with exponential penalty for the
zero-velocity gridlock crush state.

CA-CRS+(k) = w1*D_norm(k) + Phi(D,S) + w3*C_norm
Phi(D,S)   = w2*S*(1-D) + gamma*exp(lambda*(D-S))

ATTRIBUTION: Uses raw linear feature ratios (w1*D, w2*S, w3*C)
for stable, interpretable causal attribution — separate from the
non-linear scoring which captures the crush penalty. This cleanly
separates "how risky" (non-linear) from "why risky" (linear).

Calibrated parameters (tuned for correct attribution across all scenarios):
  w1=0.40, w2=0.30, w3=0.30
  gamma=0.05, lambda=4.0
  Dominance threshold: 0.40 (>40% contribution = dominant factor)
"""

import math

SAFE    = "SAFE"
WARNING = "WARNING"
DANGER  = "DANGER"

FACTOR_DENSITY  = "DENSITY"
FACTOR_SPEED    = "SPEED"
FACTOR_CONFLICT = "CONFLICT"
FACTOR_MIXED    = "MIXED"

THRESH_WARNING   = 0.35
THRESH_DANGER    = 0.70
DOMINANCE_THRESH = 0.40    # factor must contribute >40% to be called dominant


class CACRSScorer:

    def __init__(self, w1=0.40, w2=0.30, w3=0.30,
                 gamma=0.05, lam=4.0):
        assert abs(w1 + w2 + w3 - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.w1    = w1
        self.w2    = w2
        self.w3    = w3
        self.gamma = gamma
        self.lam   = lam

    def score(self, d_norm: float, s_norm: float, c_norm: float):
        """
        Returns (crs, components, phi).
        components: weighted LINEAR contributions for attribution.
        phi: full non-linear term value.
        """
        d = float(max(0.0, min(d_norm, 1.0)))
        s = float(max(0.0, min(s_norm, 1.0)))
        c = float(max(0.0, min(c_norm, 1.0)))

        phi = self.w2 * s * (1 - d) + self.gamma * math.exp(self.lam * (d - s))
        crs = min(self.w1 * d + phi + self.w3 * c, 1.0)

        # Attribution uses raw linear contributions (stable, interpretable)
        components = {
            "density":  round(self.w1 * d, 4),
            "speed":    round(self.w2 * s, 4),
            "conflict": round(self.w3 * c, 4),
        }
        return round(crs, 4), components, round(phi, 4)

    def classify(self, crs: float) -> str:
        if crs < THRESH_WARNING:
            return SAFE
        elif crs < THRESH_DANGER:
            return WARNING
        else:
            return DANGER

    def dominant_factor(self, components: dict) -> str:
        """
        Attribution from raw linear feature contributions.
        Factor is dominant if it contributes >40% of linear total.
        """
        total = sum(components.values())
        if total == 0:
            return FACTOR_MIXED
        ratios = {k: v / total for k, v in components.items()}
        best   = max(ratios, key=ratios.get)
        if ratios[best] > DOMINANCE_THRESH:
            return {
                "density":  FACTOR_DENSITY,
                "speed":    FACTOR_SPEED,
                "conflict": FACTOR_CONFLICT,
            }[best]
        return FACTOR_MIXED

    def component_ratios(self, components: dict) -> dict:
        total = sum(components.values())
        if total == 0:
            return {k: 0.0 for k in components}
        return {k: round(v / total, 3) for k, v in components.items()}