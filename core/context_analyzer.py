"""
context_analyzer.py
===================
SafeHer — Smart Context Analyzer

Analyzes ALL contextual factors:
  - Time of day risk
  - Location risk (crime zones)
  - Crowd density
  - Weather/lighting conditions
  - Day of week patterns
  - GPS speed (walking vs running vs vehicle)
  - Isolation detection
  - Known danger zones
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class ContextAnalyzer:
    """
    Analyzes environmental and situational context
    to compute a risk score (0.0 to 1.0).

    All scores normalized to 0-1.
    Combined score used as input to fusion model.
    """

    def analyze(
        self,
        hour:                 Optional[int]   = None,
        day_of_week:          Optional[int]   = None,
        gps_speed_kmh:        float           = 0.0,
        location_risk_score:  float           = 0.0,   # 0-100 from crime data
        is_isolated_area:     bool            = False,
        is_known_danger_zone: bool            = False,
        crowd_density:        float           = 0.5,   # 0=empty, 1=crowded
        is_night:             Optional[bool]  = None,
        battery_level:        float           = 100.0, # 0-100
        phone_face_down:      bool            = False,
    ) -> Dict:
        """
        Compute full context risk score.

        Returns:
        {
            "context_score": 0.0-1.0,
            "context_score_pct": 0-100,
            "risk_factors": [...],
            "breakdown": {...}
        }
        """
        if hour is None:
            hour = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        if is_night is None:
            is_night = (hour >= 20 or hour <= 6)

        # ── Individual risk components ────────────────────────────
        time_risk      = self._time_risk(hour, day_of_week)
        location_risk  = float(np.clip(location_risk_score / 100.0, 0.0, 1.0))
        isolation_risk = self._isolation_risk(is_isolated_area, crowd_density)
        zone_risk      = 0.90 if is_known_danger_zone else 0.0
        speed_risk     = self._speed_risk(gps_speed_kmh)
        battery_risk   = self._battery_risk(battery_level, is_night)

        # ── Weighted combination ──────────────────────────────────
        context_score = (
            time_risk      * 0.30 +
            location_risk  * 0.25 +
            isolation_risk * 0.20 +
            zone_risk      * 0.15 +
            speed_risk     * 0.05 +
            battery_risk   * 0.05
        )
        context_score = float(np.clip(context_score, 0.0, 1.0))

        # ── Identify active risk factors ──────────────────────────
        risk_factors = []
        if time_risk > 0.60:
            risk_factors.append("🌙 High-risk night hours")
        elif time_risk > 0.35:
            risk_factors.append("🌆 Evening risk hours")
        if is_known_danger_zone:
            risk_factors.append("🚨 Known danger zone")
        if is_isolated_area or crowd_density < 0.2:
            risk_factors.append("⚠️  Isolated area detected")
        if location_risk > 0.60:
            risk_factors.append("📍 High crime area")
        if 2.0 < gps_speed_kmh < 15.0:
            risk_factors.append("🏃 Fast movement on foot")
        if battery_level < 15 and is_night:
            risk_factors.append("🔋 Low battery at night")
        if not risk_factors:
            risk_factors.append("✅ Context normal")

        return {
            "context_score":     round(context_score, 4),
            "context_score_pct": round(context_score * 100, 1),
            "is_night":          is_night,
            "risk_factors":      risk_factors,
            "breakdown": {
                "time_risk":       round(time_risk,      3),
                "location_risk":   round(location_risk,  3),
                "isolation_risk":  round(isolation_risk, 3),
                "zone_risk":       round(zone_risk,      3),
                "speed_risk":      round(speed_risk,     3),
                "battery_risk":    round(battery_risk,   3),
            }
        }

    # ── Private risk calculators ───────────────────────────────────

    def _time_risk(self, hour: int, day_of_week: int) -> float:
        """
        Risk based on hour of day.
        Source: NCRB India — crimes against women by hour pattern.
        """
        # Night hours (highest risk)
        if 22 <= hour or hour <= 4:
            base = 0.90
        # Late evening
        elif 20 <= hour <= 22:
            base = 0.65
        # Early morning
        elif 4 <= hour <= 6:
            base = 0.55
        # Evening
        elif 18 <= hour <= 20:
            base = 0.38
        # Afternoon
        elif 12 <= hour <= 18:
            base = 0.15
        # Morning
        else:
            base = 0.12

        # Weekend nights slightly higher risk
        if day_of_week in [4, 5] and (hour >= 20 or hour <= 4):
            base = min(1.0, base + 0.08)

        return float(base)

    def _isolation_risk(self, is_isolated: bool, crowd_density: float) -> float:
        """Risk from being alone or in low-crowd areas."""
        if is_isolated:
            return 0.85
        # crowd_density: 0=empty, 1=crowded
        # Low crowd = higher risk
        crowd_risk = float(np.clip(1.0 - crowd_density, 0.0, 1.0))
        return crowd_risk * 0.70

    def _speed_risk(self, speed_kmh: float) -> float:
        """
        Risk from GPS speed pattern.
        Walking fast = possible escape.
        Very fast = possible vehicle abduction.
        """
        if 2.0 < speed_kmh <= 8.0:
            return 0.15   # brisk walking
        elif 8.0 < speed_kmh <= 15.0:
            return 0.25   # running
        elif 15.0 < speed_kmh <= 60.0:
            return 0.10   # normal vehicle
        elif speed_kmh > 60.0:
            return 0.20   # fast vehicle (possible forced)
        else:
            return 0.0    # stationary

    def _battery_risk(self, battery_level: float, is_night: bool) -> float:
        """Low battery at night = higher risk (can't call for help)."""
        if battery_level < 10 and is_night:
            return 0.80
        elif battery_level < 15 and is_night:
            return 0.60
        elif battery_level < 20 and is_night:
            return 0.35
        elif battery_level < 15:
            return 0.20
        else:
            return 0.0

    def get_time_label(self, hour: int) -> str:
        """Human readable time risk label."""
        if 22 <= hour or hour <= 4:
            return "🔴 High Risk Night Hours"
        elif 20 <= hour <= 22:
            return "🟠 Evening Risk Hours"
        elif 4 <= hour <= 6:
            return "🟡 Early Morning Risk"
        elif 18 <= hour <= 20:
            return "🟡 Late Evening"
        else:
            return "🟢 Normal Hours"