"""
movement_detector.py
====================
SafeHer — Window-Based Movement Detector

FIXED VERSION:
- Uses window of 50 samples (not single values)
- Uses central feature_extractor.py (28 features — matches trainer exactly)
- Maintains rolling buffer for continuous monitoring
- Feature mismatch bug eliminated permanently
"""

import numpy as np
import joblib
import os
import sys
import warnings
from collections import deque
from typing import Dict, Optional, List

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_extractor import (
    extract_movement_features,
    MOVEMENT_FEATURE_NAMES,
    N_MOVEMENT_FEATURES,
    validate_features,
    pad_or_trim_window,
    WINDOW_SIZE
)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_path, "models", "movement_model.pkl")


class MovementDetector:
    """
    Detects danger movement patterns using a rolling 50-sample window.

    Usage:
        detector = MovementDetector()

        # Feed sensor readings one at a time (call every ~20ms)
        detector.add_sample(ax, ay, az, gx, gy, gz)

        # Get prediction (uses last 50 samples = 1 second)
        result = detector.predict()
    """

    def __init__(self):
        # Load model
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print(f"✅ Movement model loaded: {N_MOVEMENT_FEATURES} features")
        else:
            self.model = None
            print(f"⚠️  Movement model not found. Run trainer.py first.")

        # Rolling buffers (deque auto-trims to WINDOW_SIZE)
        self._ax = deque(maxlen=WINDOW_SIZE)
        self._ay = deque(maxlen=WINDOW_SIZE)
        self._az = deque(maxlen=WINDOW_SIZE)
        self._gx = deque(maxlen=WINDOW_SIZE)
        self._gy = deque(maxlen=WINDOW_SIZE)
        self._gz = deque(maxlen=WINDOW_SIZE)

        # Prediction history for smoothing
        self._prob_history = deque(maxlen=5)

    def add_sample(self, ax: float, ay: float, az: float,
                   gx: float = 0.0, gy: float = 0.0, gz: float = 0.0):
        """Add one sensor reading to the rolling window."""
        self._ax.append(float(ax))
        self._ay.append(float(ay))
        self._az.append(float(az))
        self._gx.append(float(gx))
        self._gy.append(float(gy))
        self._gz.append(float(gz))

    def add_batch(self, samples: List[Dict]):
        """
        Add a batch of sensor readings.
        Each sample: {"ax": ..., "ay": ..., "az": ...,
                      "gx": ..., "gy": ..., "gz": ...}
        """
        for s in samples:
            self.add_sample(
                s.get("x", s.get("ax", 0.0)),
                s.get("y", s.get("ay", 0.0)),
                s.get("z", s.get("az", 0.0)),
                s.get("gx", 0.0),
                s.get("gy", 0.0),
                s.get("gz", 0.0)
            )

    def predict(self) -> Dict:
        """
        Predict danger from current window buffer.
        Returns full result dict with probability, level, scenario.
        """
        if self.model is None:
            return {"error": "Model not loaded", "status": "Failed"}

        if len(self._ax) < 10:
            return {
                "is_danger": False,
                "danger_probability": 0.0,
                "danger_level": "SAFE",
                "status": "Insufficient data",
                "samples_in_buffer": len(self._ax),
                "samples_needed": WINDOW_SIZE
            }

        # Pad if buffer not full yet
        ax = pad_or_trim_window(list(self._ax))
        ay = pad_or_trim_window(list(self._ay))
        az = pad_or_trim_window(list(self._az))
        gx = pad_or_trim_window(list(self._gx))
        gy = pad_or_trim_window(list(self._gy))
        gz = pad_or_trim_window(list(self._gz))

        features = extract_movement_features(ax, ay, az, gx, gy, gz)

        if not validate_features(features, N_MOVEMENT_FEATURES, "movement"):
            return {"error": "Feature extraction failed", "status": "Failed"}

        try:
            prediction    = self.model.predict([features])[0]
            proba         = self.model.predict_proba([features])[0]
            danger_prob   = round(float(proba[1]) * 100, 1)
            is_danger     = bool(prediction == 1)

            # Smooth with history (reduces jitter from single spikes)
            self._prob_history.append(danger_prob)
            smoothed_prob = round(float(np.mean(self._prob_history)), 1)

            danger_level, action = self._get_danger_level(smoothed_prob)
            scenario             = self._identify_scenario(features, is_danger)
            alert_triggered      = is_danger and smoothed_prob >= 60.0

            # Sensor summary for UI display
            accel_mag = float(np.sqrt(features[0]**2))  # mean accel
            gyro_mag  = float(features[10])              # mean gyro
            impact    = float(features[20])              # impact_force
            shake     = float(features[19])              # phone_shake

            return {
                "is_danger":          is_danger,
                "danger_probability": smoothed_prob,
                "raw_probability":    danger_prob,
                "danger_level":       danger_level,
                "scenario_detected":  scenario,
                "action":             action,
                "alert_triggered":    alert_triggered,
                "status":             "Danger Movement" if is_danger else "Normal Movement",
                "features_used":      N_MOVEMENT_FEATURES,
                "buffer_size":        len(self._ax),
                "sensor_summary": {
                    "accel_mean":   round(features[0], 3),
                    "accel_peak":   round(features[2], 3),
                    "gyro_mean":    round(features[10], 3),
                    "jerk_mean":    round(features[4], 3),
                    "impact_force": round(impact, 3),
                    "phone_shake":  round(shake, 3)
                }
            }

        except Exception as e:
            return {"error": str(e), "status": "Processing Failed"}

    def predict_from_lists(
        self,
        ax_window: List[float], ay_window: List[float], az_window: List[float],
        gx_window: List[float] = None, gy_window: List[float] = None,
        gz_window: List[float] = None
    ) -> Dict:
        """
        Predict directly from lists (for API calls from Flutter).
        Flutter sends 50 samples → this function handles them directly.
        """
        if gx_window is None:
            gx_window = [0.0] * len(ax_window)
            gy_window = [0.0] * len(ax_window)
            gz_window = [0.0] * len(ax_window)

        # Add to buffer and predict
        for i in range(len(ax_window)):
            self.add_sample(
                ax_window[i], ay_window[i], az_window[i],
                gx_window[i], gy_window[i], gz_window[i]
            )

        return self.predict()

    def clear_buffer(self):
        """Reset rolling window (call when user pauses monitoring)."""
        self._ax.clear(); self._ay.clear(); self._az.clear()
        self._gx.clear(); self._gy.clear(); self._gz.clear()
        self._prob_history.clear()

    # ── Private helpers ────────────────────────────────────────────

    def _get_danger_level(self, probability: float):
        if probability >= 90:
            return "CRITICAL DANGER", "🔴 Immediate SOS Required"
        elif probability >= 75:
            return "HIGH DANGER",     "🟠 Dangerous Movement — Alert Sent"
        elif probability >= 60:
            return "MODERATE DANGER", "🟡 Abnormal Movement — Monitoring"
        elif probability >= 40:
            return "LOW RISK",        "🟢 Slight Anomaly — Watching"
        else:
            return "SAFE",            "✅ Normal Movement"

    def _identify_scenario(self, features: np.ndarray, is_danger: bool) -> str:
        """Identify specific scenario from feature values."""
        impact    = features[20]   # impact_force
        jerk      = features[4]    # accel_jerk_mean
        jerk_peak = features[5]    # accel_jerk_peak
        gyro_mean = features[10]   # gyro_mean
        gyro_std  = features[11]   # gyro_std
        accel_m   = features[0]    # accel_mean
        shake     = features[19]   # phone_shake
        entropy   = features[25]   # signal_entropy

        if is_danger:
            if impact > 25 and jerk_peak > 20:
                return "🚨 Kidnapping / Violent Attack"
            elif accel_m > 22 and jerk > 10:
                return "🚨 Physical Assault"
            elif impact > 18 and accel_m > 14:
                return "🚨 Sudden Fall / Push"
            elif impact > 25 and shake > 15:
                return "⚠️  Phone Snatched"
            elif accel_m > 12 and gyro_mean > 8:
                return "⚠️  Escape / Panic Running"
            elif gyro_std > 5 and accel_m < 12:
                return "⚠️  Trembling / Fear"
            elif entropy > 3.5:
                return "⚠️  Struggle / Restrained"
            else:
                return "⚠️  Harassment / Being Followed"
        else:
            if accel_m < 1.5:
                return "✅ Sitting / Still"
            elif accel_m < 6:
                return "✅ Normal Walking"
            elif accel_m < 14:
                return "✅ Jogging / Exercise"
            else:
                return "✅ Normal Commute / Travel"