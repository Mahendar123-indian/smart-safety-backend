"""
main.py
=======
SafeHer ML Backend — FastAPI Server v3.0

Endpoints:
  POST /analyze_danger         → Full 7-factor danger analysis (JSON sensors)
  POST /predict/audio_file     → Audio file upload → scream detection
  POST /predict/movement       → Movement window prediction
  GET  /baseline/stats/{uid}   → Behavioral baseline progress
  POST /baseline/update/{uid}  → Update user's baseline
  WS   /ws/monitor/{uid}       → WebSocket real-time monitoring
  GET  /health                 → Server + model status
  GET  /model/info             → Model details

FIXES in this version:
  - Audio endpoint accepts file upload (not 7 scalars)
  - Movement uses 28-feature window extractor (no more mismatch)
  - Unified 7-factor weights from danger_scorer.py
  - WebSocket for continuous real-time streaming from Flutter
  - Behavioral baseline integration
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import logging
import traceback
import tempfile
from datetime import datetime
from typing import Optional, List

import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.feature_extractor import (
    extract_movement_features,
    N_MOVEMENT_FEATURES,
    N_AUDIO_FEATURES,
    MOVEMENT_FEATURE_NAMES,
    validate_features,
    pad_or_trim_window,
    WINDOW_SIZE
)
from core.movement_detector  import MovementDetector
from core.audio_processor    import ScreamDetector
from core.danger_scorer      import DangerScorer, WEIGHTS, THRESHOLDS
from core.behavioral_baseline import BaselineManager

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SafeHer")

# ── Model paths ──────────────────────────────────────────────────
MODELS_DIR          = os.path.join(BASE_DIR, "models")
MOVEMENT_MODEL_PATH = os.path.join(MODELS_DIR, "movement_model.pkl")
SCREAM_MODEL_PATH   = os.path.join(MODELS_DIR, "scream_model.pkl")
DANGER_MODEL_PATH   = os.path.join(MODELS_DIR, "danger_model.pkl")

# ── Global instances ─────────────────────────────────────────────
movement_detector = None
scream_detector   = None
danger_model      = None
danger_scorer     = DangerScorer()
models_loaded     = {}


def load_models():
    global movement_detector, scream_detector, danger_model, models_loaded

    try:
        movement_detector  = MovementDetector()
        models_loaded["movement"] = movement_detector.model is not None
    except Exception as e:
        log.warning(f"⚠️  Movement detector failed: {e}")
        models_loaded["movement"] = False

    try:
        scream_detector = ScreamDetector()
        models_loaded["scream"] = scream_detector.model is not None
    except Exception as e:
        log.warning(f"⚠️  Scream detector failed: {e}")
        models_loaded["scream"] = False

    try:
        danger_model = joblib.load(DANGER_MODEL_PATH)
        models_loaded["danger"] = True
        log.info("✅ Danger fusion model loaded")
    except Exception as e:
        log.warning(f"⚠️  Danger model not loaded: {e}")
        models_loaded["danger"] = False

    loaded = sum(1 for v in models_loaded.values() if v)
    log.info(f"{'✅' if loaded == 3 else '⚠️'} {loaded}/3 models loaded")


# ══════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════

class SensorSample(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class GyroSample(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class DangerRequest(BaseModel):
    user_id:              str   = "default"
    # Movement — send 50 samples (1 second of data)
    accel_samples:        List[SensorSample] = Field(default_factory=list)
    gyro_samples:         List[SensorSample] = Field(default_factory=list)
    # Audio scalar (from client-side audio processing)
    audio_scream_probability: float = 0.0   # 0-100
    # Context
    hour_of_day:          int   = 12
    gps_lat:              float = 0.0
    gps_lon:              float = 0.0
    gps_speed_kmh:        float = 0.0
    location_risk_score:  float = 0.0    # 0-100
    is_known_danger_zone: bool  = False
    is_isolated_area:     bool  = False
    # Phone behavior
    phone_face_down:      bool  = False
    phone_shake:          float = 0.0
    impact_force:         float = 0.0
    # Route
    route_deviation:      float = 0.0    # 0-100
    timestamp:            Optional[int] = None

class MovementRequest(BaseModel):
    user_id:       str  = "default"
    accel_samples: List[SensorSample] = Field(default_factory=list)
    gyro_samples:  List[SensorSample] = Field(default_factory=list)

class BaselineUpdateRequest(BaseModel):
    accel_samples: List[SensorSample] = Field(default_factory=list)
    gyro_samples:  List[SensorSample] = Field(default_factory=list)
    hour:          int = 12
    day_of_week:   int = 0
    gps_lat:       float = 0.0
    gps_lon:       float = 0.0

class DangerResponse(BaseModel):
    danger_score:           float
    danger_level:           str
    movement_probability:   float
    audio_probability:      float
    behavioral_deviation:   float
    context_score:          float
    confidence:             float
    trigger_recommendation: str
    sos_triggered:          bool
    insights:               List[str]
    score_breakdown:        dict
    processing_ms:          int
    models_used:            List[str]
    timestamp:              str


# ══════════════════════════════════════════════════════════════════
# CORE PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _movement_from_request(accel: List[SensorSample],
                            gyro:  List[SensorSample]) -> tuple:
    """Extract movement probability from sensor samples."""
    if not accel:
        return 0.0, False

    ax = pad_or_trim_window([s.x for s in accel])
    ay = pad_or_trim_window([s.y for s in accel])
    az = pad_or_trim_window([s.z for s in accel])
    gx = pad_or_trim_window([s.x for s in gyro] if gyro else [0.0]*50)
    gy = pad_or_trim_window([s.y for s in gyro] if gyro else [0.0]*50)
    gz = pad_or_trim_window([s.z for s in gyro] if gyro else [0.0]*50)

    if movement_detector and movement_detector.model:
        result = movement_detector.predict_from_lists(ax, ay, az, gx, gy, gz)
        return result.get("danger_probability", 0.0), True

    # Heuristic fallback
    features = extract_movement_features(ax, ay, az, gx, gy, gz)
    impact   = features[20]
    jerk     = features[4]
    shake    = features[19]
    p = float(np.clip(impact/30*0.4 + jerk/15*0.3 + shake/20*0.3, 0, 1)) * 100
    return p, False


def _predict_danger_fusion(
    movement_prob: float, audio_prob: float, behavioral_dev: float,
    context_score: float, location_risk: float,
    phone_behavior: float, route_deviation: float
) -> tuple:
    """Run the 7-factor danger fusion model."""
    features = np.array([[
        movement_prob/100, audio_prob/100, behavioral_dev/100,
        context_score/100, location_risk/100,
        phone_behavior/100, route_deviation/100
    ]])

    if models_loaded.get("danger") and danger_model is not None:
        try:
            if hasattr(danger_model, "predict_proba"):
                score = float(danger_model.predict_proba(features)[0][1])
            else:
                score = float(danger_model.predict(features)[0])
            return float(np.clip(score * 100, 0, 100)), True
        except Exception as e:
            log.warning(f"Danger model predict failed: {e}")

    # Weighted fallback using unified WEIGHTS
    score = (
        WEIGHTS["movement"]             * movement_prob    +
        WEIGHTS["audio"]                * audio_prob       +
        WEIGHTS["behavioral_deviation"] * behavioral_dev   +
        WEIGHTS["context"]              * context_score    +
        WEIGHTS["location_risk"]        * location_risk    +
        WEIGHTS["phone_behavior"]       * phone_behavior   +
        WEIGHTS["route_deviation"]      * route_deviation
    )
    return float(np.clip(score, 0, 100)), False


def _build_insights(level: str, movement: float, audio: float,
                     behavioral: float, hour: int, speed: float) -> List[str]:
    insights = []
    if movement > 70:
        insights.append("🚨 Violent motion pattern detected by movement AI.")
    elif movement > 45:
        insights.append(f"⚠️  Unusual movement detected ({movement:.0f}% danger).")
    else:
        insights.append("✅ Movement pattern normal.")

    if audio > 70:
        insights.append("🔊 Distress audio detected — scream classifier triggered.")
    elif audio > 40:
        insights.append(f"🔉 Elevated audio distress ({audio:.0f}%).")
    else:
        insights.append("🔇 Audio environment normal.")

    if behavioral > 60:
        insights.append("🧠 Behavior deviates significantly from your personal baseline.")

    if 22 <= hour or hour <= 5:
        insights.append("🌙 Night hours — AI monitoring sensitivity increased.")

    if 2.0 < speed < 15.0:
        insights.append("🏃 Fast movement — tracking your path.")

    if level == "SOS":
        insights.append("🚨 All emergency contacts being notified NOW.")
        insights.append("🎥 Evidence recording active.")
    elif level == "DANGER":
        insights.append("📲 Tap SOS if you feel unsafe. Auto-alert in 10s.")
    elif level == "WARNING":
        insights.append("📍 Move to a well-lit, populated area.")
    elif level == "SAFE":
        insights.append("✅ Stay safe. Emergency contacts are ready.")

    return insights[:6]


# ══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SafeHer ML Backend",
    description="7-Factor AI danger detection for women's safety",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    log.info("=" * 55)
    log.info("  SafeHer ML Backend v3.0 starting...")
    log.info("=" * 55)
    load_models()
    log.info("✅ Server ready. Waiting for Flutter connections...")


# ── Health check ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":         "healthy",
        "server":         "SafeHer ML Backend v3.0",
        "models_loaded":  models_loaded,
        "all_models_ok":  all(models_loaded.values()) if models_loaded else False,
        "features": {
            "movement": N_MOVEMENT_FEATURES,
            "audio":    N_AUDIO_FEATURES,
            "fusion":   7
        },
        "weights":        WEIGHTS,
        "timestamp":      datetime.now().isoformat()
    }


# ── Main danger analysis ──────────────────────────────────────────

@app.post("/analyze_danger", response_model=DangerResponse)
def analyze_danger(req: DangerRequest):
    t_start = time.monotonic()
    try:
        # 1. Movement prediction (28 features, window-based)
        movement_prob, mv_used_model = _movement_from_request(
            req.accel_samples, req.gyro_samples
        )

        # 2. Audio probability (from client-side processing)
        audio_prob = float(np.clip(req.audio_scream_probability, 0, 100))

        # 3. Behavioral deviation
        behavioral_dev = 0.0
        if req.accel_samples and len(req.accel_samples) >= 10:
            ax = [s.x for s in req.accel_samples]
            ay = [s.y for s in req.accel_samples]
            az = [s.z for s in req.accel_samples]
            gx = [s.x for s in req.gyro_samples] if req.gyro_samples else [0]*len(ax)
            gy = [s.y for s in req.gyro_samples] if req.gyro_samples else [0]*len(ax)
            gz = [s.z for s in req.gyro_samples] if req.gyro_samples else [0]*len(ax)

            ax_w = pad_or_trim_window(ax)
            ay_w = pad_or_trim_window(ay)
            az_w = pad_or_trim_window(az)
            gx_w = pad_or_trim_window(gx)
            gy_w = pad_or_trim_window(gy)
            gz_w = pad_or_trim_window(gz)

            features = extract_movement_features(ax_w, ay_w, az_w, gx_w, gy_w, gz_w)
            now = datetime.now()
            dev_result = BaselineManager.get_deviation(
                req.user_id, features, req.hour_of_day, now.weekday()
            )
            behavioral_dev = float(dev_result.get("deviation_score", 0.0) * 100)

            # Update baseline with current reading
            BaselineManager.update_all(
                req.user_id, features, req.hour_of_day, now.weekday()
            )

        # 4. Context score
        context_result = danger_scorer._context_score(
            req.hour_of_day, req.is_isolated_area,
            req.is_known_danger_zone, req.gps_speed_kmh
        )
        context_score = float(context_result * 100)

        # 5. Phone behavior
        pb = danger_scorer._phone_behavior_score(
            req.phone_shake, req.impact_force, req.phone_face_down
        )
        phone_behavior = float(pb * 100)

        # 6. Final fusion (7 factors)
        danger_score_100, fusion_used = _predict_danger_fusion(
            movement_prob, audio_prob, behavioral_dev,
            context_score, req.location_risk_score,
            phone_behavior, req.route_deviation
        )

        # 7. Level + action
        score_01 = danger_score_100 / 100.0
        if   score_01 < THRESHOLDS["ALERT"][0]:    level, trigger = "SAFE",    "none"
        elif score_01 < THRESHOLDS["WARNING"][0]:  level, trigger = "ALERT",   "monitor"
        elif score_01 < THRESHOLDS["DANGER"][0]:   level, trigger = "WARNING", "countdown_10s"
        elif score_01 < THRESHOLDS["SOS"][0]:      level, trigger = "DANGER",  "countdown_5s"
        else:                                       level, trigger = "SOS",     "immediate_sos"

        sos_triggered = level == "SOS"

        # 8. Confidence
        n_models = sum([mv_used_model, True, fusion_used])
        confidence = round(0.50 + (n_models / 3) * 0.45, 4)

        insights = _build_insights(
            level, movement_prob, audio_prob,
            behavioral_dev, req.hour_of_day, req.gps_speed_kmh
        )

        models_used = []
        if mv_used_model:  models_used.append("movement_rf_28feat")
        models_used.append("scream_client_side")
        if fusion_used:    models_used.append("danger_fusion_7factor")
        if not models_used: models_used.append("heuristic_fallback")

        ms = int((time.monotonic() - t_start) * 1000)
        log.info(f"[analyze_danger] score={danger_score_100:.1f} "
                 f"level={level} {ms}ms uid={req.user_id}")

        return DangerResponse(
            danger_score=          round(danger_score_100, 2),
            danger_level=          level,
            movement_probability=  round(movement_prob, 2),
            audio_probability=     round(audio_prob, 2),
            behavioral_deviation=  round(behavioral_dev, 2),
            context_score=         round(context_score, 2),
            confidence=            confidence,
            trigger_recommendation=trigger,
            sos_triggered=         sos_triggered,
            insights=              insights,
            score_breakdown={
                "movement":           round(movement_prob * WEIGHTS["movement"], 2),
                "audio":              round(audio_prob * WEIGHTS["audio"], 2),
                "behavioral":         round(behavioral_dev * WEIGHTS["behavioral_deviation"], 2),
                "context":            round(context_score * WEIGHTS["context"], 2),
                "location":           round(req.location_risk_score * WEIGHTS["location_risk"], 2),
                "phone_behavior":     round(phone_behavior * WEIGHTS["phone_behavior"], 2),
                "route":              round(req.route_deviation * WEIGHTS["route_deviation"], 2),
            },
            processing_ms=         ms,
            models_used=           models_used,
            timestamp=             datetime.now().isoformat()
        )

    except Exception as e:
        log.error(f"[analyze_danger] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Audio file upload endpoint ────────────────────────────────────

@app.post("/predict/audio_file")
async def predict_audio_file(file: UploadFile = File(...)):
    """
    Accept audio file upload from Flutter → extract 230 features →
    run scream model → return result.

    Flutter usage:
        var request = http.MultipartRequest('POST', uri);
        request.files.add(await http.MultipartFile.fromPath('file', audioPath));
    """
    t_start = time.monotonic()
    tmp_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content  = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        if scream_detector is None:
            raise HTTPException(status_code=503, detail="Audio model not loaded")

        result = scream_detector.analyze_audio(tmp_path)
        result["processing_ms"] = int((time.monotonic() - t_start) * 1000)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Movement-only endpoint ────────────────────────────────────────

@app.post("/predict/movement")
def predict_movement(req: MovementRequest):
    t_start = time.monotonic()
    prob, used = _movement_from_request(req.accel_samples, req.gyro_samples)
    return {
        "movement_probability": round(prob, 2),
        "is_dangerous":         prob > 60.0,
        "confidence":           0.88 if used else 0.55,
        "model_used":           "movement_gb_28feat" if used else "heuristic",
        "features_count":       N_MOVEMENT_FEATURES,
        "processing_ms":        int((time.monotonic() - t_start) * 1000)
    }


# ── Behavioral baseline endpoints ─────────────────────────────────

@app.get("/baseline/stats/{user_id}")
def baseline_stats(user_id: str):
    baseline = BaselineManager.get(user_id)
    return baseline.get_stats()


@app.post("/baseline/update/{user_id}")
def baseline_update(user_id: str, req: BaselineUpdateRequest):
    if not req.accel_samples:
        raise HTTPException(status_code=400, detail="accel_samples required")

    ax = pad_or_trim_window([s.x for s in req.accel_samples])
    ay = pad_or_trim_window([s.y for s in req.accel_samples])
    az = pad_or_trim_window([s.z for s in req.accel_samples])
    gx = pad_or_trim_window([s.x for s in req.gyro_samples] if req.gyro_samples else [0]*50)
    gy = pad_or_trim_window([s.y for s in req.gyro_samples] if req.gyro_samples else [0]*50)
    gz = pad_or_trim_window([s.z for s in req.gyro_samples] if req.gyro_samples else [0]*50)

    features = extract_movement_features(ax, ay, az, gx, gy, gz)
    baseline = BaselineManager.get(user_id)
    baseline.update(features, req.hour, req.day_of_week)

    if req.gps_lat != 0.0 and req.gps_lon != 0.0:
        baseline.update_location(req.gps_lat, req.gps_lon,
                                  req.hour, req.day_of_week)

    return {"status": "updated", "stats": baseline.get_stats()}


# ── WebSocket real-time monitoring ─────────────────────────────────

@app.websocket("/ws/monitor/{user_id}")
async def websocket_monitor(websocket: WebSocket, user_id: str):
    """
    Real-time WebSocket for continuous monitoring from Flutter.

    Flutter sends JSON every 2 seconds:
    {
        "accel_samples": [{"x":0.1,"y":-9.8,"z":0.2}, ...],
        "gyro_samples":  [...],
        "audio_scream_probability": 0.0,
        "hour_of_day": 22,
        "gps_speed_kmh": 0.0,
        "location_risk_score": 30.0,
        "phone_face_down": false
    }
    """
    await websocket.accept()
    log.info(f"[WS] Connected: {user_id}")

    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)

            # Build DangerRequest from incoming JSON
            req = DangerRequest(
                user_id=              user_id,
                accel_samples=        [SensorSample(**s) for s in data.get("accel_samples", [])],
                gyro_samples=         [SensorSample(**s) for s in data.get("gyro_samples", [])],
                audio_scream_probability= float(data.get("audio_scream_probability", 0)),
                hour_of_day=          int(data.get("hour_of_day", 12)),
                gps_speed_kmh=        float(data.get("gps_speed_kmh", 0)),
                location_risk_score=  float(data.get("location_risk_score", 0)),
                is_known_danger_zone= bool(data.get("is_known_danger_zone", False)),
                is_isolated_area=     bool(data.get("is_isolated_area", False)),
                phone_face_down=      bool(data.get("phone_face_down", False)),
                phone_shake=          float(data.get("phone_shake", 0)),
                impact_force=         float(data.get("impact_force", 0)),
                route_deviation=      float(data.get("route_deviation", 0)),
            )

            # Run analysis (reuse HTTP handler logic)
            result = analyze_danger(req)
            await websocket.send_json(result.dict())

    except WebSocketDisconnect:
        log.info(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        log.error(f"[WS] Error for {user_id}: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ── Model info ────────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
    info = {
        "models":      {},
        "weights":     WEIGHTS,
        "thresholds":  THRESHOLDS,
        "features": {
            "movement": N_MOVEMENT_FEATURES,
            "audio":    N_AUDIO_FEATURES,
            "fusion":   7
        },
        "server_time": datetime.now().isoformat()
    }
    if models_loaded.get("movement") and movement_detector and movement_detector.model:
        info["models"]["movement"] = {
            "type":       type(movement_detector.model).__name__,
            "n_features": N_MOVEMENT_FEATURES,
            "status":     "loaded"
        }
    if models_loaded.get("scream") and scream_detector and scream_detector.model:
        info["models"]["scream"] = {
            "type":       type(scream_detector.model).__name__,
            "n_features": N_AUDIO_FEATURES,
            "status":     "loaded"
        }
    if models_loaded.get("danger"):
        info["models"]["danger"] = {
            "type":       type(danger_model).__name__ if danger_model else "N/A",
            "n_features": 7,
            "status":     "loaded"
        }
    return info


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  SafeHer ML Backend v3.0")
    print("  Listening on: http://0.0.0.0:8001")
    print("  WebSocket:    ws://0.0.0.0:8001/ws/monitor/{user_id}")
    print("═" * 55 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8001,
                reload=True, log_level="info")