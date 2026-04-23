"""
simulation.py
─────────────
Projects post-intervention CA-CRS+ score using empirical
reduction coefficients applied component-wise.

The non-linear scoring formula is preserved in the projection
so the exponential crush penalty is correctly modulated.
"""

import math
from modules.risk_scoring import CACRSScorer, DANGER, THRESH_DANGER

# Reduction coefficients calibrated from Helbing et al. (2007) and
# Oliveira (2025) crowd dynamics literature.
XI = {
    "OPEN":     {"d": 0.60, "s": 0.85, "c": 0.90},
    "CLOSE":    {"d": 0.80, "s": 0.55, "c": 0.80},
    "REDIRECT": {"d": 0.85, "s": 0.80, "c": 0.40},
    "HOLD":     {"d": 0.95, "s": 0.95, "c": 0.95},
}


class InterventionSimulator:

    def __init__(self, w1=0.40, w2=0.30, w3=0.30,
                 gamma=0.05, lam=4.0):
        # Keep projection scoring identical to the main CA-CRS+ scorer.
        # Otherwise a reduced post-action state can appear riskier simply
        # because projection used harsher gamma/lambda parameters.
        self.scorer = CACRSScorer(w1, w2, w3, gamma, lam)

    def project(self, d_norm: float, s_norm: float,
                c_norm: float, action: str) -> float:
        """
        Project CA-CRS+ after applying gate action.

        Returns projected CRS score.
        """
        xi = XI.get(action, XI["HOLD"])
        d_p = d_norm * xi["d"]
        s_p = s_norm * xi["s"]
        c_p = c_norm * xi["c"]
        proj_crs, _, _ = self.scorer.score(d_p, s_p, c_p)
        return round(proj_crs, 4)

    def reduction_rate(self, crs_before: float,
                       crs_proj: float) -> float:
        """CRS Reduction Rate (%) = 100*(1 - CRS_proj/CRS_before)"""
        if crs_before == 0:
            return 0.0
        return round(100 * (1 - crs_proj / crs_before), 1)

    def compare_all_actions(self, d_norm: float, s_norm: float,
                            c_norm: float, current_crs: float) -> dict:
        """
        Rank all gate actions by projected CRS (ascending).
        Used for fallback chain if primary action insufficient.

        Returns OrderedDict: {action: (proj_crs, crr)}
        """
        from collections import OrderedDict
        results = {}
        for action in XI:
            proj = self.project(d_norm, s_norm, c_norm, action)
            crr  = self.reduction_rate(current_crs, proj)
            results[action] = {"proj_crs": proj, "crr": crr}
        return OrderedDict(
            sorted(results.items(), key=lambda x: x[1]["proj_crs"])
        )

    def validate_action(self, d_norm: float, s_norm: float,
                        c_norm: float, action: str,
                        current_crs: float) -> dict:
        """
        Validate recommended action. If projected CRS remains at DANGER,
        return the next-best action automatically.
        """
        proj = self.project(d_norm, s_norm, c_norm, action)
        if proj < THRESH_DANGER:
            return {
                "action":    action,
                "proj_crs":  proj,
                "crr":       self.reduction_rate(current_crs, proj),
                "escalated": False,
            }
        # Fallback: try all actions
        ranked = self.compare_all_actions(
            d_norm, s_norm, c_norm, current_crs
        )
        for fallback_action, metrics in ranked.items():
            if metrics["proj_crs"] < THRESH_DANGER:
                return {
                    "action":    fallback_action,
                    "proj_crs":  metrics["proj_crs"],
                    "crr":       metrics["crr"],
                    "escalated": True,
                    "note":      f"Original action '{action}' insufficient; "
                                 f"escalated to '{fallback_action}'",
                }
        # All actions insufficient — escalate to human
        return {
            "action":    "ESCALATE",
            "proj_crs":  proj,
            "crr":       0.0,
            "escalated": True,
            "note":      "All actions insufficient. Human supervisor required.",
        }
