"""
behavioral_baseline.py
======================
SafeHer — Personal Behavioral Baseline Engine

Learns each user's PERSONAL normal movement pattern over 7-14 days.
Danger = deviation from HER own pattern, not a generic threshold.

This is the most powerful accuracy booster because:
- A nurse walking fast at 3AM is normal for her
- A student sitting alone at 11PM is normal for her
- Only deviations from personal baseline trigger alerts

Storage: Firebase Firestore (user_id -> baseline profile)
Local fallback: JSON file per user
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings("ignore")

BASELINE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines")
os.makedirs(BASELINE_DIR, exist_ok=True)

MIN_SAMPLES_FOR_BASELINE = 500      # minimum readings before baseline is reliable
BASELINE_WINDOW_DAYS     = 14       # learn from last 14 days
DEVIATION_ALERT_THRESHOLD = 2.5     # standard deviations from personal mean
EXTREME_DEVIATION_THRESHOLD = 4.0   # extreme anomaly threshold


class BehavioralBaseline:
    """
    Maintains a rolling statistical baseline of user's normal behavior.

    Features tracked per time-slot (hour of day × day of week):
    - accel_mean distribution
    - gyro_mean distribution
    - speed distribution
    - typical locations (cluster centers)
    - phone usage patterns
    """

    def __init__(self, user_id: str = "default"):
        self.user_id    = user_id
        self.profile    = self._load_profile()
        self._buffer    = deque(maxlen=200)   # recent readings buffer

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def update(self, features: np.ndarray, hour: int, day_of_week: int):
        """
        Add a new reading to the baseline.
        Call this every time sensor data is collected (every 2 seconds).
        """
        slot = self._time_slot(hour, day_of_week)

        if slot not in self.profile["slots"]:
            self.profile["slots"][slot] = {
                "count": 0,
                "mean":  np.zeros(len(features)).tolist(),
                "M2":    np.zeros(len(features)).tolist(),   # Welford's algorithm
                "min":   features.tolist(),
                "max":   features.tolist()
            }

        self._welford_update(slot, features)
        self.profile["total_samples"] += 1
        self._buffer.append(features.tolist())

        # Auto-save every 100 updates
        if self.profile["total_samples"] % 100 == 0:
            self._save_profile()

    def get_deviation_score(
        self,
        features: np.ndarray,
        hour: int,
        day_of_week: int
    ) -> Dict:
        """
        Calculate how much this reading deviates from personal baseline.

        Returns:
            {
                "deviation_score": 0.0-1.0,
                "is_anomaly": bool,
                "confidence": 0.0-1.0,
                "details": {...}
            }
        """
        if self.profile["total_samples"] < MIN_SAMPLES_FOR_BASELINE:
            # Not enough data yet — use global fallback
            return {
                "deviation_score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "details": {"reason": "insufficient_baseline_data",
                            "samples_collected": self.profile["total_samples"],
                            "samples_needed": MIN_SAMPLES_FOR_BASELINE}
            }

        slot = self._time_slot(hour, day_of_week)

        # Fallback to nearest available slot if current slot has no data
        slot_data = self._get_slot_data(slot)

        if slot_data is None or slot_data["count"] < 10:
            return {
                "deviation_score": 0.0,
                "is_anomaly": False,
                "confidence": 0.1,
                "details": {"reason": "no_slot_data"}
            }

        mean  = np.array(slot_data["mean"])
        count = slot_data["count"]
        M2    = np.array(slot_data["M2"])

        # Welford variance
        variance = M2 / max(count - 1, 1)
        std      = np.sqrt(variance + 1e-10)

        # Z-score for each feature
        z_scores = np.abs((features - mean) / std)

        # Weight key danger-indicative features more
        weights = self._feature_weights()
        weighted_z = z_scores * weights

        # Composite deviation score
        mean_z      = float(np.mean(weighted_z))
        max_z       = float(np.max(weighted_z))
        top5_z      = float(np.mean(np.sort(weighted_z)[-5:]))

        # Normalize to 0-1
        deviation_score = float(np.clip(
            (0.4 * mean_z + 0.3 * max_z + 0.3 * top5_z) / EXTREME_DEVIATION_THRESHOLD,
            0.0, 1.0
        ))

        is_anomaly = (mean_z > DEVIATION_ALERT_THRESHOLD or
                      max_z > EXTREME_DEVIATION_THRESHOLD)

        # Confidence grows with number of baseline samples
        confidence = float(min(1.0, count / 1000.0))

        return {
            "deviation_score": round(deviation_score, 4),
            "is_anomaly":      is_anomaly,
            "confidence":      round(confidence, 4),
            "details": {
                "mean_z_score":  round(mean_z, 3),
                "max_z_score":   round(max_z, 3),
                "slot":          slot,
                "slot_samples":  count,
                "total_samples": self.profile["total_samples"]
            }
        }

    def get_route_deviation(
        self,
        lat: float,
        lon: float,
        hour: int,
        day_of_week: int
    ) -> float:
        """
        Check if current location deviates from usual routes at this time.
        Returns 0.0 (normal) to 1.0 (completely unknown location).
        """
        slot = self._time_slot(hour, day_of_week)
        locations = self.profile.get("locations", {}).get(slot, [])

        if not locations:
            return 0.3   # unknown slot → mild concern, not alarm

        # Minimum distance to any known location in this time slot
        min_dist = min(
            self._haversine(lat, lon, loc[0], loc[1])
            for loc in locations
        )

        # >500m from any known location at this time = unusual
        route_deviation = float(np.clip(min_dist / 500.0, 0.0, 1.0))
        return route_deviation

    def update_location(self, lat: float, lon: float, hour: int, day_of_week: int):
        """Record a location visit for route learning."""
        slot = self._time_slot(hour, day_of_week)

        if "locations" not in self.profile:
            self.profile["locations"] = {}

        if slot not in self.profile["locations"]:
            self.profile["locations"][slot] = []

        # Only add if >50m from existing known locations (avoid duplicates)
        existing = self.profile["locations"][slot]
        if existing:
            min_dist = min(self._haversine(lat, lon, loc[0], loc[1]) for loc in existing)
            if min_dist < 50:
                return  # too close to existing — skip

        self.profile["locations"][slot].append([lat, lon])

        # Keep max 20 locations per slot
        if len(self.profile["locations"][slot]) > 20:
            self.profile["locations"][slot] = self.profile["locations"][slot][-20:]

    def is_baseline_ready(self) -> bool:
        return self.profile["total_samples"] >= MIN_SAMPLES_FOR_BASELINE

    def get_stats(self) -> Dict:
        return {
            "user_id":        self.user_id,
            "total_samples":  self.profile["total_samples"],
            "slots_learned":  len(self.profile["slots"]),
            "baseline_ready": self.is_baseline_ready(),
            "progress_pct":   round(min(100.0, self.profile["total_samples"] /
                                        MIN_SAMPLES_FOR_BASELINE * 100), 1)
        }

    # ──────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────

    def _welford_update(self, slot: str, features: np.ndarray):
        """Welford's online algorithm for running mean and variance."""
        s = self.profile["slots"][slot]
        s["count"] += 1
        n = s["count"]

        mean = np.array(s["mean"])
        M2   = np.array(s["M2"])

        delta  = features - mean
        mean   = mean + delta / n
        delta2 = features - mean
        M2     = M2 + delta * delta2

        s["mean"] = mean.tolist()
        s["M2"]   = M2.tolist()
        s["min"]  = np.minimum(np.array(s["min"]), features).tolist()
        s["max"]  = np.maximum(np.array(s["max"]), features).tolist()

    def _get_slot_data(self, slot: str):
        """Get slot data, falling back to adjacent slots if needed."""
        if slot in self.profile["slots"] and self.profile["slots"][slot]["count"] >= 10:
            return self.profile["slots"][slot]

        # Try adjacent hour slots
        parts = slot.split("_")
        if len(parts) == 2:
            try:
                hour = int(parts[0])
                day  = int(parts[1])
                for delta in [1, -1, 2, -2]:
                    adj_slot = self._time_slot((hour + delta) % 24, day)
                    if adj_slot in self.profile["slots"] and \
                       self.profile["slots"][adj_slot]["count"] >= 10:
                        return self.profile["slots"][adj_slot]
            except Exception:
                pass
        return None

    def _feature_weights(self) -> np.ndarray:
        """
        Higher weights for features most indicative of danger.
        Aligned with MOVEMENT_FEATURE_NAMES order (28 features).
        """
        weights = np.ones(28)
        # Boost danger-critical features
        weights[4]  = 2.0   # accel_jerk_mean
        weights[5]  = 2.5   # accel_jerk_peak
        weights[6]  = 1.5   # gravity_deviation
        weights[14] = 2.0   # gyro_jerk_mean
        weights[15] = 2.5   # gyro_jerk_peak
        weights[18] = 1.5   # combined_intensity
        weights[19] = 2.0   # phone_shake
        weights[20] = 2.5   # impact_force
        weights[25] = 1.5   # signal_entropy
        return weights / weights.sum() * len(weights)  # normalize

    @staticmethod
    def _time_slot(hour: int, day_of_week: int) -> str:
        return f"{hour}_{day_of_week}"

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Distance in meters between two GPS coordinates."""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi  = np.radians(lat2 - lat1)
        dlam  = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        return float(2 * R * np.arcsin(np.sqrt(a)))

    def _load_profile(self) -> Dict:
        path = os.path.join(BASELINE_DIR, f"{self.user_id}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "user_id":       self.user_id,
            "total_samples": 0,
            "created_at":    datetime.now().isoformat(),
            "slots":         {},
            "locations":     {}
        }

    def _save_profile(self):
        path = os.path.join(BASELINE_DIR, f"{self.user_id}.json")
        try:
            with open(path, "w") as f:
                json.dump(self.profile, f)
        except Exception as e:
            print(f"⚠️  Failed to save baseline: {e}")


# ══════════════════════════════════════════════════════════════════
# BASELINE MANAGER — manages multiple user baselines in memory
# ══════════════════════════════════════════════════════════════════

class BaselineManager:
    """Singleton manager for all active user baselines."""

    _instances: Dict[str, BehavioralBaseline] = {}

    @classmethod
    def get(cls, user_id: str) -> BehavioralBaseline:
        if user_id not in cls._instances:
            cls._instances[user_id] = BehavioralBaseline(user_id)
        return cls._instances[user_id]

    @classmethod
    def update_all(cls, user_id: str, features: np.ndarray,
                   hour: int, day_of_week: int):
        baseline = cls.get(user_id)
        baseline.update(features, hour, day_of_week)

    @classmethod
    def get_deviation(cls, user_id: str, features: np.ndarray,
                      hour: int, day_of_week: int) -> Dict:
        baseline = cls.get(user_id)
        return baseline.get_deviation_score(features, hour, day_of_week)