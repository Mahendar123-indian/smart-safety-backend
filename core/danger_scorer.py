"""
danger_scorer.py
================
SafeHer — Unified 7-Factor Danger Scorer

FIXED VERSION:
- Single consistent weight system (used everywhere)
- 7 factors: movement, audio, behavioral deviation,
              context, location, phone behavior, route
- Confidence intervals for each prediction
- 5-stage alert system
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional


# ══════════════════════════════════════════════════════════════════
# UNIFIED WEIGHTS — used in BOTH danger_scorer.py AND main.py
# Change here → automatically applies everywhere
# ══════════════════════════════════════════════════════════════════

WEIGHTS = {
    "movement":            0.28,   # physical movement patterns
    "audio":               0.25,   # scream/distress audio
    "behavioral_deviation":0.18,   # personal baseline anomaly
    "context":             0.12,   # time of day + night risk
    "location_risk":       0.10,   # crime area risk
    "phone_behavior":      0.04,   # phone drop, face down, silence
    "route_deviation":     0.03    # unusual location/route
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# 5-stage alert thresholds
THRESHOLDS = {
    "SAFE":     (0.00, 0.30),
    "ALERT":    (0.30, 0.48),
    "WARNING":  (0.48, 0.62),
    "DANGER":   (0.62, 0.78),
    "SOS":      (0.78, 1.00)
}


class DangerScorer:
    """
    Combines all ML model outputs into a final unified danger score.

    All weights are defined in WEIGHTS dict above.
    Consistent with train_danger_model.py feature order.
    """

    def calculate_danger_score(
        self,
        movement_probability:    float = 0.0,   # 0-100 from MovementDetector
        audio_probability:       float = 0.0,   # 0-100 from ScreamDetector
        behavioral_deviation:    float = 0.0,   # 0-100 from BehavioralBaseline
        hour:                    Optional[int] = None,
        is_isolated_area:        bool  = False,
        location_risk_score:     float = 0.0,   # 0-100
        phone_shake:             float = 0.0,
        impact_force:            float = 0.0,
        phone_face_down:         bool  = False,
        is_known_danger_zone:    bool  = False,
        route_deviation:         float = 0.0,   # 0-100
        gps_speed_kmh:           float = 0.0
    ) -> Dict:
        """
        Calculate unified danger score from all factors.
        All probability inputs are 0-100 (normalized to 0-1 internally).
        """
        if hour is None:
            hour = datetime.now().hour

        # Normalize all inputs to 0-1
        mv  = float(np.clip(movement_probability    / 100.0, 0.0, 1.0))
        au  = float(np.clip(audio_probability        / 100.0, 0.0, 1.0))
        bd  = float(np.clip(behavioral_deviation     / 100.0, 0.0, 1.0))
        lr  = float(np.clip(location_risk_score      / 100.0, 0.0, 1.0))
        rd  = float(np.clip(route_deviation          / 100.0, 0.0, 1.0))

        # Context score (time + isolation + known zone)
        cx  = self._context_score(hour, is_isolated_area, is_known_danger_zone,
                                   gps_speed_kmh)

        # Phone behavior score
        pb  = self._phone_behavior_score(phone_shake, impact_force, phone_face_down)

        # Weighted fusion
        score = (
            WEIGHTS["movement"]             * mv +
            WEIGHTS["audio"]                * au +
            WEIGHTS["behavioral_deviation"] * bd +
            WEIGHTS["context"]              * cx +
            WEIGHTS["location_risk"]        * lr +
            WEIGHTS["phone_behavior"]       * pb +
            WEIGHTS["route_deviation"]      * rd
        )
        score = float(np.clip(score, 0.0, 1.0))

        # Determine level and action
        level, action, sos = self._classify(score)

        # Confidence based on how many signals are active
        active_signals = sum([mv > 0.2, au > 0.2, bd > 0.2, cx > 0.3, lr > 0.2])
        confidence = float(np.clip(0.5 + active_signals * 0.10, 0.5, 1.0))

        return {
            "danger_score":         round(score * 100, 1),        # 0-100 for UI
            "danger_score_raw":     round(score, 4),               # 0-1 for model
            "danger_level":         level,
            "action":               action,
            "sos_triggered":        sos,
            "confidence":           round(confidence, 3),
            "score_breakdown": {
                "movement_score":    round(mv  * WEIGHTS["movement"]             * 100, 1),
                "audio_score":       round(au  * WEIGHTS["audio"]                * 100, 1),
                "behavioral_score":  round(bd  * WEIGHTS["behavioral_deviation"] * 100, 1),
                "context_score":     round(cx  * WEIGHTS["context"]              * 100, 1),
                "location_score":    round(lr  * WEIGHTS["location_risk"]        * 100, 1),
                "phone_score":       round(pb  * WEIGHTS["phone_behavior"]       * 100, 1),
                "route_score":       round(rd  * WEIGHTS["route_deviation"]      * 100, 1),
            },
            "individual_values": {
                "movement_probability":    round(mv  * 100, 1),
                "audio_probability":       round(au  * 100, 1),
                "behavioral_deviation":    round(bd  * 100, 1),
                "context_risk":            round(cx  * 100, 1),
                "location_risk":           round(lr  * 100, 1),
                "phone_behavior":          round(pb  * 100, 1),
                "route_deviation":         round(rd  * 100, 1),
            }
        }

    # ── Private helpers ────────────────────────────────────────────

    def _context_score(self, hour: int, is_isolated: bool,
                        is_danger_zone: bool, gps_speed: float) -> float:
        """Risk score from time + location context."""
        # Time of day risk (NCRB India — crimes against women by hour)
        if   22 <= hour or hour <= 4:   time_risk = 0.90
        elif 20 <= hour <= 22:          time_risk = 0.65
        elif 4  <= hour <= 6:           time_risk = 0.55
        elif 18 <= hour <= 20:          time_risk = 0.40
        else:                           time_risk = 0.15

        isolation_risk = 0.80 if is_isolated   else 0.10
        zone_risk      = 0.90 if is_danger_zone else 0.10

        # GPS speed: very fast (being driven) or walking speed
        speed_risk = 0.0
        if 2.0 < gps_speed < 15.0:    speed_risk = 0.15   # on foot fast
        elif gps_speed > 60.0:         speed_risk = 0.20   # vehicle

        score = (time_risk * 0.45 + isolation_risk * 0.25 +
                 zone_risk  * 0.20 + speed_risk     * 0.10)
        return float(np.clip(score, 0.0, 1.0))

    def _phone_behavior_score(self, shake: float, impact: float,
                                face_down: bool) -> float:
        """Risk from phone physical behavior."""
        shake_score    = float(np.clip(shake * 0.04, 0.0, 1.0))
        impact_score   = float(np.clip(impact * 0.025, 0.0, 1.0))
        face_down_score = 0.35 if face_down else 0.0
        return float(np.clip(shake_score * 0.35 + impact_score * 0.35 +
                              face_down_score * 0.30, 0.0, 1.0))

    @staticmethod
    def _classify(score: float):
        if score < THRESHOLDS["ALERT"][0]:
            return "SAFE",    "✅ All Clear — Monitoring Active",          False
        elif score < THRESHOLDS["WARNING"][0]:
            return "ALERT",   "🟡 Elevated Risk — Watching Closely",       False
        elif score < THRESHOLDS["DANGER"][0]:
            return "WARNING", "🟠 Warning — Ready to Alert Contacts",      False
        elif score < THRESHOLDS["SOS"][0]:
            return "DANGER",  "🔴 Danger Detected — Countdown Started",    False
        else:
            return "SOS",     "🚨 SOS TRIGGERED — Contacts Notified",      True