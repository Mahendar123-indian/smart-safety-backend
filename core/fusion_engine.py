"""
fusion_engine.py
================
SafeHer — Advanced Fusion Engine

Combines ALL model outputs into one final danger decision:
  - Movement model probability
  - Audio scream probability
  - Behavioral baseline deviation
  - Context score (time + location)
  - Location risk score
  - Phone behavior score
  - Route deviation score

Uses trained danger_model.pkl for fusion.
Falls back to weighted scoring if model not loaded.
"""

import os
import sys
import numpy as np
import joblib
import warnings
from datetime import datetime
from typing import Dict, Optional, List

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.danger_scorer import DangerScorer, WEIGHTS, THRESHOLDS
from core.behavioral_baseline import BaselineManager

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANGER_MODEL_PATH = os.path.join(BASE_DIR, "models", "danger_model.pkl")


class FusionEngine:
    """
    Final fusion layer — combines all signals into one danger score.

    Usage:
        engine = FusionEngine()
        result = engine.predict(
            user_id            = "user123",
            movement_prob      = 75.0,
            audio_prob         = 60.0,
            hour               = 22,
            location_risk      = 40.0,
            is_isolated        = True,
            phone_face_down    = False,
            gps_speed_kmh      = 5.0,
            route_deviation    = 20.0
        )
    """

    def __init__(self):
        self.scorer = DangerScorer()
        self._load_model()

    def _load_model(self):
        if os.path.exists(DANGER_MODEL_PATH):
            try:
                self.model = joblib.load(DANGER_MODEL_PATH)
                print("✅ Fusion danger model loaded!")
            except Exception as e:
                self.model = None
                print(f"⚠️  Fusion model load failed: {e}")
        else:
            self.model = None
            print("⚠️  danger_model.pkl not found. Run train_danger_model.py")

    def predict(
        self,
        user_id:              str   = "default",
        movement_prob:        float = 0.0,    # 0-100
        audio_prob:           float = 0.0,    # 0-100
        hour:                 Optional[int] = None,
        gps_lat:              float = 0.0,
        gps_lon:              float = 0.0,
        gps_speed_kmh:        float = 0.0,
        location_risk:        float = 0.0,    # 0-100
        is_isolated_area:     bool  = False,
        is_known_danger_zone: bool  = False,
        phone_face_down:      bool  = False,
        phone_shake:          float = 0.0,
        impact_force:         float = 0.0,
        route_deviation:      float = 0.0,    # 0-100
        movement_features:    Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run full 7-factor danger fusion.

        Returns complete result dict with:
        - danger_score (0-100)
        - danger_level (SAFE/ALERT/WARNING/DANGER/SOS)
        - sos_triggered (bool)
        - score_breakdown (per factor)
        - trigger_recommendation
        - insights
        """
        if hour is None:
            hour = datetime.now().hour

        now = datetime.now()

        # ── Step 1: Behavioral deviation ─────────────────────────
        behavioral_dev = 0.0
        if movement_features is not None:
            try:
                dev_result     = BaselineManager.get_deviation(
                    user_id, movement_features, hour, now.weekday()
                )
                behavioral_dev = float(dev_result.get("deviation_score", 0.0) * 100)

                # Update baseline with current reading
                BaselineManager.update_all(
                    user_id, movement_features, hour, now.weekday()
                )

                # Update location if available
                if gps_lat != 0.0 and gps_lon != 0.0:
                    baseline = BaselineManager.get(user_id)
                    baseline.update_location(gps_lat, gps_lon, hour, now.weekday())

            except Exception as e:
                print(f"⚠️  Baseline deviation error: {e}")

        # ── Step 2: Context score ─────────────────────────────────
        context_score = float(
            self.scorer._context_score(
                hour, is_isolated_area, is_known_danger_zone, gps_speed_kmh
            ) * 100
        )

        # ── Step 3: Phone behavior score ──────────────────────────
        phone_behavior = float(
            self.scorer._phone_behavior_score(
                phone_shake, impact_force, phone_face_down
            ) * 100
        )

        # ── Step 4: Route deviation ───────────────────────────────
        if movement_features is not None and gps_lat != 0.0 and gps_lon != 0.0:
            try:
                baseline       = BaselineManager.get(user_id)
                route_deviation = float(
                    baseline.get_route_deviation(gps_lat, gps_lon, hour, now.weekday()) * 100
                )
            except Exception:
                pass  # keep passed route_deviation value

        # ── Step 5: Fusion model or weighted fallback ─────────────
        danger_score, used_model = self._fuse(
            movement_prob, audio_prob, behavioral_dev,
            context_score, location_risk, phone_behavior, route_deviation
        )

        # ── Step 6: Classify ──────────────────────────────────────
        score_01 = danger_score / 100.0
        if   score_01 < THRESHOLDS["ALERT"][0]:   level, trigger = "SAFE",    "none"
        elif score_01 < THRESHOLDS["WARNING"][0]: level, trigger = "ALERT",   "monitor"
        elif score_01 < THRESHOLDS["DANGER"][0]:  level, trigger = "WARNING", "countdown_10s"
        elif score_01 < THRESHOLDS["SOS"][0]:     level, trigger = "DANGER",  "countdown_5s"
        else:                                      level, trigger = "SOS",     "immediate_sos"

        sos_triggered = level == "SOS"

        # ── Step 7: Confidence ────────────────────────────────────
        active = sum([
            movement_prob > 20,
            audio_prob    > 20,
            behavioral_dev > 20,
            context_score  > 30,
            location_risk  > 20
        ])
        confidence = round(min(1.0, 0.50 + active * 0.10), 3)

        # ── Step 8: Insights ──────────────────────────────────────
        insights = self._build_insights(
            level, movement_prob, audio_prob,
            behavioral_dev, context_score, hour
        )

        return {
            "danger_score":           round(danger_score, 2),
            "danger_level":           level,
            "sos_triggered":          sos_triggered,
            "trigger_recommendation": trigger,
            "confidence":             confidence,
            "model_used":             "danger_fusion_7factor" if used_model else "weighted_fallback",
            "score_breakdown": {
                "movement":    round(movement_prob  * WEIGHTS["movement"],             2),
                "audio":       round(audio_prob     * WEIGHTS["audio"],                2),
                "behavioral":  round(behavioral_dev * WEIGHTS["behavioral_deviation"], 2),
                "context":     round(context_score  * WEIGHTS["context"],              2),
                "location":    round(location_risk  * WEIGHTS["location_risk"],        2),
                "phone":       round(phone_behavior * WEIGHTS["phone_behavior"],       2),
                "route":       round(route_deviation* WEIGHTS["route_deviation"],      2),
            },
            "individual_scores": {
                "movement_probability": round(movement_prob,   2),
                "audio_probability":    round(audio_prob,      2),
                "behavioral_deviation": round(behavioral_dev,  2),
                "context_score":        round(context_score,   2),
                "location_risk":        round(location_risk,   2),
                "phone_behavior":       round(phone_behavior,  2),
                "route_deviation":      round(route_deviation, 2),
            },
            "insights":   insights,
            "timestamp":  datetime.now().isoformat()
        }

    def _fuse(
        self,
        movement: float, audio: float, behavioral: float,
        context: float, location: float,
        phone: float, route: float
    ):
        """Run fusion model or fall back to weighted scoring."""
        features = np.array([[
            movement  / 100, audio    / 100, behavioral / 100,
            context   / 100, location / 100, phone      / 100,
            route     / 100
        ]])

        if self.model is not None:
            try:
                if hasattr(self.model, "predict_proba"):
                    score = float(self.model.predict_proba(features)[0][1]) * 100
                else:
                    score = float(self.model.predict(features)[0]) * 100
                return float(np.clip(score, 0, 100)), True
            except Exception as e:
                print(f"⚠️  Fusion model predict failed: {e}")

        # Weighted fallback
        score = (
            WEIGHTS["movement"]             * movement   +
            WEIGHTS["audio"]                * audio      +
            WEIGHTS["behavioral_deviation"] * behavioral +
            WEIGHTS["context"]              * context    +
            WEIGHTS["location_risk"]        * location   +
            WEIGHTS["phone_behavior"]       * phone      +
            WEIGHTS["route_deviation"]      * route
        )
        return float(np.clip(score, 0, 100)), False

    @staticmethod
    def _build_insights(
        level: str,
        movement: float, audio: float,
        behavioral: float, context: float,
        hour: int
    ) -> List[str]:
        insights = []

        if movement > 70:
            insights.append("🚨 Violent motion pattern detected.")
        elif movement > 45:
            insights.append(f"⚠️  Unusual movement detected ({movement:.0f}%).")
        else:
            insights.append("✅ Movement pattern normal.")

        if audio > 70:
            insights.append("🔊 Distress audio detected.")
        elif audio > 40:
            insights.append(f"🔉 Elevated audio distress ({audio:.0f}%).")
        else:
            insights.append("🔇 Audio normal.")

        if behavioral > 60:
            insights.append("🧠 Behavior deviates from your personal baseline.")

        if 22 <= hour or hour <= 5:
            insights.append("🌙 Night hours — monitoring sensitivity increased.")

        if level == "SOS":
            insights.append("🚨 Emergency contacts being notified NOW.")
        elif level == "DANGER":
            insights.append("📲 Tap SOS if unsafe. Auto-alert in 10s.")
        elif level == "WARNING":
            insights.append("📍 Move to a well-lit populated area.")

        return insights[:5]